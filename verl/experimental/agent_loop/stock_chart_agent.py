"""
verl/trainer/ppo/rollout/agent_loop/stock_chart_agent.py

Multi-turn agent loop for the stock candlestick identification task.

The model is given a screenshot of a stock chart and must identify the
6-digit stock code.  It may call three external tools during reasoning:

    FinQuery    — query financial data (stock codes, prices, …)
    Search      — web/news search
    TickerChart — fetch a K-line chart image for a given stock & date range

The agent follows the Hermes tool-call format:
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>

and ends its turn with:
    <FINISHED>
    股票代码：XXXXXX

At that point the loop terminates and the calc_stock_reward tool scores the answer.

State machine
-------------
  PENDING            → tokenise prompt
  GENERATING         → run LLM; check for <tool_call> or <FINISHED>
  PROCESSING_TOOLS   → execute FinQuery / Search / TickerChart calls
  SCORING            → call calc_stock_reward with the model's final answer
  TERMINATED         → done
"""


# ── DEV ONLY: fake model output for tool-call testing ──────────────


# fake_text = (
#         "Thought: 我需要先通过金融查询工具获取截图中可能出现的股票代码。\n"
#         "<tool_call>"
#         '{"name": "FinQuery", "arguments": {"query": "同花顺的股票代码"}}'
#         "</tool_call>"
#     )

# fake_text = (
#         "Thought: 我再搜索一下相关新闻来辅助判断。\n"
#         "<tool_call>"
#         '{"name": "Search", "arguments": {"query": "同花顺 300033 最新动态"}}'
#         "</tool_call>"
#     )

fake_text = (
        "Thought: 现在获取候选股票的K线图与截图对比。\n"
        "<tool_call>"
        '{"name": "TickerChart", "arguments": {'
        '"codeName": "300033", "chartType": "Daily Candlestick", '
        '"startDate": "2025-08-28", "endDate": "2025-10-21", '
        '"indicator": ["MA", "MACD"]}}'
        "</tool_call>"
    )

# fake_text = (
#         "Thought: 信息完整，我已确定股票代码。\n"
#         "<FINISHED>\n"
#         "股票代码：300033"
#     )


import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
import re
# ---------------------------------------------------------------------------
# Additional state for this agent
# ---------------------------------------------------------------------------



class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


class StockAgentState(Enum):
    PENDING          = "pending"
    GENERATING       = "generating"
    PROCESSING_TOOLS = "processing_tools"
    SCORING          = "scoring"       # dedicated state: call calc_stock_reward
    TERMINATED       = "terminated"


# Regex to detect the model's <FINISHED> signal and capture its answer
_FINISHED_RE = re.compile(
    r"<FINISHED>.*?股票代码[：:]\s*(\d{6})",
    re.DOTALL | re.IGNORECASE,
)
# Looser fallback: any 6-digit number that appears after <FINISHED>
_FINISHED_LOOSE_RE = re.compile(r"<FINISHED>.*?(\d{6})", re.DOTALL)


def _extract_final_answer(text: str) -> str | None:
    """Pull the 6-digit code from the model's <FINISHED> block."""
    m = _FINISHED_RE.search(text)
    if m:
        return m.group(1)
    m = _FINISHED_LOOSE_RE.search(text)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@register("stock_chart_agent")
class StockChartAgentLoop(AgentLoopBase):
    """
    Multi-turn RL agent loop for stock chart identification.

    Differences from the generic ToolAgentLoop:
      * Adds a dedicated SCORING state that fires calc_stock_reward when
        the model outputs <FINISHED>.
      * Injects tool-call results that may include images (TickerChart)
        back into the conversation as multi-modal content.
      * Enforces a hard cap on external tool calls (max_tool_calls) to
        prevent runaway API usage during rollout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mt = self.rollout_config.multi_turn
        self.max_user_turns          = mt.max_user_turns
        self.max_assistant_turns     = mt.max_assistant_turns
        self.max_parallel_calls      = mt.max_parallel_calls
        self.max_tool_response_length = mt.max_tool_response_length
        self.tool_response_truncate_side = mt.tool_response_truncate_side
        self.prompt_length           = self.rollout_config.prompt_length
        self.response_length         = self.rollout_config.response_length

        # External tools: FinQuery, Search, TickerChart
        tool_config_path = mt.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools       = {t.name: t for t in tool_list}
        self.tool_schemas = [
            t.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
            for t in tool_list
        ]

        self.tool_parser      = ToolParser.get_tool_parser(mt.format, self.tokenizer)
        self.tool_parser_name = mt.format
        

    def _log_turn_debug(
            self,
            agent_data: AgentData,
            turn_type: str,
            prompt_ids: list[int],
            response_ids: list[int] | None = None,
            tool_calls=None,
            tool_responses=None,
        ):
        """Print raw debug info for a single turn."""
        turn_num = agent_data.assistant_turns + agent_data.user_turns
        sep = "=" * 80
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        lines = [
            f"\n{sep}",
            f"[TURN DEBUG] request_id={agent_data.request_id}  turn={turn_num}  type={turn_type}",
            f"--- PROMPT ({len(prompt_ids)} tokens) ---",
            prompt_text,
        ]
        # lines = []
        if response_ids is not None:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            lines += [
                f"--- RESPONSE ({len(response_ids)} tokens) ---",
                response_text,
            ]
        if tool_calls:
            lines += [
                "--- TOOL CALLS ---",
                *[f"  [{i}] {tc.name}({tc.arguments})" for i, tc in enumerate(tool_calls)],
            ]
        if tool_responses:
            lines += [
                "--- TOOL RESPONSES ---",
                *[f"  [{i}] {r}" for i, r in enumerate(tool_responses)],
            ]
        lines.append(sep)
        print("\n".join(lines), flush=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        request_id   = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics={},
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        # ── initialise the reward tool instance ──────────────────────────
        reward_tool   = self.tools.get("calc_stock_reward")
        reward_instance_id = None
        if reward_tool is not None:
            rw_kwargs = tools_kwargs.get("calc_stock_reward", {})
            reward_instance_id, _ = await reward_tool.create(
                **rw_kwargs.get("create_kwargs", {})
            )

        # ── state machine ────────────────────────────────────────────────
        state = StockAgentState.PENDING
        while state != StockAgentState.TERMINATED:
            if state == StockAgentState.PENDING:
                state = await self._handle_pending(agent_data, sampling_params)
            elif state == StockAgentState.GENERATING:
                state = await self._handle_generating(
                    agent_data, sampling_params, reward_instance_id
                )
            elif state == StockAgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools(agent_data)
            elif state == StockAgentState.SCORING:
                state = await self._handle_scoring(
                    agent_data, reward_tool, reward_instance_id
                )
            else:
                logger.error(f"Unknown state {state}, terminating.")
                state = StockAgentState.TERMINATED

        # ── release reward tool ──────────────────────────────────────────
        if reward_tool is not None and reward_instance_id is not None:
            await reward_tool.release(reward_instance_id)

        # ── build output ─────────────────────────────────────────────────
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids   = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        mm_data = {}
        if agent_data.image_data is not None:
            mm_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            mm_data["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=mm_data,
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
        })
        return output

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _handle_pending(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> StockAgentState:
        """Tokenise the initial prompt (system + user message with image)."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return StockAgentState.GENERATING

    async def _handle_generating(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        reward_instance_id: str | None,
    ) -> StockAgentState:
        """Run the LLM for one turn and decide the next state."""
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
    
            # import os, types
            # fake_ids = self.tokenizer.encode(fake_text, add_special_tokens=False)
            # output = types.SimpleNamespace(
            #         token_ids=fake_ids,
            #         log_probs=None,
            #         num_preempted=0,
            #         routed_experts=None,
            #         extra_fields={},
            #     )


        # ✅ Log after generation
        # self._log_turn_debug(
        #     agent_data,
        #     turn_type="ASSISTANT_GENERATE",
        #     prompt_ids=agent_data.prompt_ids[: -len(output.token_ids)],  # prompt before appending response
        #     response_ids=output.token_ids,
        #     tool_calls=agent_data.tool_calls if agent_data.tool_calls else None,
        # )
        # ── book-keeping ─────────────────────────────────────────────────
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = (
                output.num_preempted if output.num_preempted is not None else -1
            )
        else:
            agent_data.metrics["num_preempted"] += (
                output.num_preempted if output.num_preempted is not None else 0
            )

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_gs = output.extra_fields.get("max_global_steps")
            if max_gs:
                agent_data.extra_fields["max_global_steps"] = max_gs

        agent_data.assistant_turns += 1
        agent_data.response_ids     = output.token_ids
        agent_data.prompt_ids      += agent_data.response_ids
        agent_data.response_mask   += [1] * len(agent_data.response_ids)

        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # ── hard termination guards ───────────────────────────────────────
        if len(agent_data.response_mask) >= self.response_length:
            return StockAgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return StockAgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return StockAgentState.TERMINATED

        # ── decode response to inspect content ───────────────────────────
        response_text: str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=False),
        )

        # ── check for <FINISHED> ──────────────────────────────────────────
        if "<FINISHED>" in response_text:
            # Record assistant message before scoring
            agent_data.messages.append({"role": "assistant", "content": response_text})
            return StockAgentState.SCORING

        # ── check for tool calls ──────────────────────────────────────────
        tools = [t.tool_schema for t in self.tools.values()
                 if t.name != "calc_stock_reward"]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            output.token_ids, tools
        )

        if agent_data.tool_calls:
            agent_data.messages.append({"role": "assistant", "content": response_text})
            return StockAgentState.PROCESSING_TOOLS

        # No tool calls and no <FINISHED>: the model is mid-thought; keep generating
        # (This handles the case where the model fills the context with reasoning text.)
        agent_data.messages.append({"role": "assistant", "content": response_text})
        return StockAgentState.TERMINATED

    async def _handle_processing_tools(self, agent_data: AgentData) -> StockAgentState:
        """
        Execute FinQuery / Search / TickerChart tool calls in parallel,
        inject results (including images) back into the conversation.
        """
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        # Fire external tool calls (skip calc_stock_reward here — that has its own state)
        tasks = []
        tool_call_names = []
        for tc in agent_data.tool_calls[: self.max_parallel_calls]:
            if tc.name == "calc_stock_reward":
                # Should not appear here, but guard just in case
                continue
            tasks.append(self._call_external_tool(tc, agent_data))
            tool_call_names.append(tc.name)

        if not tasks:
            # Nothing to execute (all calls were filtered out)
            return StockAgentState.GENERATING

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
            
    #     self._log_turn_debug(
    #     agent_data,
    #     turn_type="TOOL_RESPONSE",
    #     prompt_ids=agent_data.prompt_ids,  # full prompt so far
    #     tool_responses=[r[0].text for r in responses],  # ToolResponse.text per call
    # )

        for tool_response, tool_reward in responses:
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            # Build the tool-result message
            if tool_response.image:
                content = []
                content.append({"type": "image"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
                # Collect new images for multi-modal prompt update
                imgs = tool_response.image if isinstance(tool_response.image, list) else [tool_response.image]
                new_images_this_turn.extend(i for i in imgs if i is not None)
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

        agent_data.messages.extend(add_messages)

        # Tokenise the tool-result messages
        images  = new_images_this_turn if new_images_this_turn else None
        response_ids = await self.apply_chat_template(
            add_messages,
            images=images,
            videos=None,
            remove_system_prompt=True,
        )

        # Respect response length budget
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return StockAgentState.TERMINATED

        # Update image data
        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)

        agent_data.prompt_ids    += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        agent_data.user_turns += 1
        return StockAgentState.GENERATING

    async def _handle_scoring(
        self,
        agent_data: AgentData,
        reward_tool,
        reward_instance_id: str | None,
    ) -> StockAgentState:
        """
        The model has emitted <FINISHED>.  Extract its answer from the last
        assistant message and call calc_stock_reward to get the final reward.
        """
        # Find the model's stated answer in the last assistant message
        last_assistant_text = ""
        for msg in reversed(agent_data.messages):
            if msg.get("role") == "assistant":
                last_assistant_text = msg.get("content", "")
                break

        final_answer = _extract_final_answer(last_assistant_text) or ""

        if reward_tool is not None and reward_instance_id is not None:
            try:
                tool_response, tool_reward, _ = await reward_tool.execute(
                    reward_instance_id,
                    {"answer": final_answer},
                )
                agent_data.tool_rewards.append(tool_reward)

                # Inject the scoring feedback as a system note so the model can
                # (in future turns) see whether it was right — but here we
                # immediately terminate so it serves as logged context only.
                feedback_message = {
                    "role": "tool",
                    "content": tool_response.text or "",
                }
                agent_data.messages.append(feedback_message)

                # Tokenise feedback and append (mask = 0, not trained on)
                feedback_ids = await self.apply_chat_template(
                    [feedback_message],
                    remove_system_prompt=True,
                )
                if len(agent_data.response_mask) + len(feedback_ids) < self.response_length:
                    agent_data.prompt_ids    += feedback_ids
                    agent_data.response_mask += [0] * len(feedback_ids)
                    if agent_data.response_logprobs:
                        agent_data.response_logprobs += [0.0] * len(feedback_ids)

                agent_data.turn_scores.append(tool_reward)

            except Exception as e:
                logger.warning(f"Scoring failed for request {agent_data.request_id}: {e}")

        return StockAgentState.TERMINATED

    # ------------------------------------------------------------------
    # External tool dispatch  (FinQuery / Search / TickerChart)
    # ------------------------------------------------------------------

    async def _call_external_tool(
        self,
        tool_call: FunctionCall,
        agent_data: AgentData,
    ) -> tuple[ToolResponse, float | None]:
        """
        Dispatch a single FinQuery / Search / TickerChart call.

        Returns (ToolResponse, optional_reward).
        """
        tool_name = tool_call.name
        tool = self.tools.get(tool_name)
        if tool is None:
            return (
                ToolResponse(text=f"未知工具：{tool_name}。可用工具：{list(self.tools.keys())}"),
                None,
            )

        instance_id = None
        try:
            tool_args = json.loads(tool_call.arguments)
            kwargs    = agent_data.tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(
                create_kwargs=kwargs.get("create_kwargs", {})
            )
            tool_response, tool_reward, _ = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"External tool '{tool_name}' error: {e}")
            return ToolResponse(text=f"工具调用失败 ({tool_name}): {e}"), None
        finally:
            if tool is not None and instance_id is not None:
                await tool.release(instance_id)

        # Truncate over-long responses
        text = tool_response.text or ""
        if len(text) > self.max_tool_response_length:
            side = self.tool_response_truncate_side
            L    = self.max_tool_response_length
            if side == "left":
                text = text[:L] + "...(truncated)"
            elif side == "right":
                text = "(truncated)..." + text[-L:]
            else:
                half = L // 2
                text = text[:half] + "...(truncated)..." + text[-half:]

        # Rebuild response with possibly-truncated text but preserve image/video
        kw = {"text": text}
        for attr in ("image", "video"):
            val = getattr(tool_response, attr, None)
            if val is not None:
                kw[attr] = val

        return ToolResponse(**kw), tool_reward