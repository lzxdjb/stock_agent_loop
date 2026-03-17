"""
verl/tools/stock_chart_tool.py

Tool for scoring the model's stock code identification answer.

Flow:
  1. create()   — called once per trajectory; stores the ground truth 6-digit code.
  2. execute()  — called each time the model fires <tool_call>calc_stock_reward</tool_call>;
                  parses the model's answer, computes reward, returns feedback text.
  3. calc_reward() — pure scoring logic, also used by the RL trainer at the end of rollout.
  4. release()  — cleans up per-trajectory state.

Reward scheme
-------------
  +1.0   exact match on the 6-digit code
   0.0   wrong code but a valid 6-digit number was submitted
  -0.1   answer could not be parsed as a 6-digit number

A small penalty (-0.05) is applied at execute() time if the model submits the
same answer it already submitted before (no improvement), mirroring the gsm8k tool.
"""

import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool, ToolResponse
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op


def _parse_stock_code(raw: str) -> str | None:
    """
    Extract the first 6-digit sequence from the model's answer string.

    Accepts formats like:
        "001209"
        "#### 001209"
        "股票代码：001209"
        "the stock code is 001209.SZ"
    Returns the bare 6-digit string, or None if nothing parseable is found.
    """
    # Strip common prefixes the model might emit
    raw = raw.strip()
    # Remove market suffixes (.SH / .SZ / .BJ)
    raw = re.sub(r"\.(SH|SZ|BJ|sh|sz|bj)$", "", raw.strip())
    # Find first run of exactly 6 digits
    match = re.search(r"\b(\d{6})\b", raw)
    if match:
        return match.group(1)
    # Looser fallback: any 6 consecutive digits
    match = re.search(r"(\d{6})", raw)
    if match:
        return match.group(1)
    return None


class StockChartTool(BaseTool):
    """
    Reward tool for the stock candlestick identification task.

    Tool schema (also defined in tools_config.yaml):
        name: calc_stock_reward
        parameters:
            answer (string): The model's identified 6-digit stock code.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # Maps instance_id -> per-trajectory state dict
        self._instance_dict: dict[str, dict[str, Any]] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """
        Initialise a trajectory slot.

        ground_truth can be passed either as a direct kwarg (preferred) or
        nested inside create_kwargs (as the pipeline does via tools_kwargs).
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Support both direct kwarg and the pipeline's create_kwargs nesting
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)

        if ground_truth is None:
            raise ValueError(
                "StockChartTool.create() requires 'ground_truth' (6-digit stock code)."
            )

        self._instance_dict[instance_id] = {
            "ground_truth": str(ground_truth).strip(),
            "last_answer": None,   # last parsed code the model submitted
            "best_reward": 0.0,    # best reward seen so far this trajectory
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """
        Score the model's submitted answer and return feedback.

        parameters["answer"]: the model's stock code string.

        Returns:
            tool_response  — text feedback shown back to the model
            tool_reward    — incremental reward signal for this tool call
            res            — auxiliary info dict (empty here)
        """
        state = self._instance_dict[instance_id]
        raw_answer = parameters.get("answer", "")
        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)

        parsed_code = _parse_stock_code(raw_answer)

        # ------------------------------------------------------------------
        # Build human-readable feedback
        # ------------------------------------------------------------------
        if parsed_code is None:
            feedback = (
                f"无法从您的回答 '{raw_answer}' 中解析出有效的6位股票代码。"
                f"请确保您提交的是纯数字的6位股票代码，例如 '001209'。"
            )
            reward = -0.1
            tool_reward = -0.1
            state["last_answer"] = raw_answer
            return ToolResponse(text=feedback), tool_reward, {}

        # Compute absolute reward for this submission
        reward = await self.calc_reward(instance_id, parsed_code=parsed_code)

        # Incremental tool reward: penalise if no improvement over previous best
        if reward > state["best_reward"]:
            tool_reward = reward - state["best_reward"]   # positive delta
            state["best_reward"] = reward
        else:
            tool_reward = -0.05   # penalty for redundant / regressive submission

        state["last_answer"] = parsed_code

        ground_truth = state["ground_truth"]
        if reward == 1.0:
            feedback = (
                f"✓ 正确！您提交的股票代码 {parsed_code} 与目标股票代码 {ground_truth} 完全匹配。"
            )
        else:
            feedback = (
                f"✗ 错误。您提交的股票代码为 {parsed_code}，目标股票代码为 {ground_truth}。"
                f"请重新分析图表特征后再试。"
            )

        return ToolResponse(text=feedback), tool_reward, {}

    async def calc_reward(
        self,
        instance_id: str,
        parsed_code: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Pure reward calculation.  Called by the RL trainer at trajectory end
        (via the reward_model pipeline) as well as internally by execute().

        If parsed_code is provided directly (internal call from execute),
        use it; otherwise fall back to the last submitted answer stored in state.
        """
        state = self._instance_dict[instance_id]
        ground_truth = state["ground_truth"]

        if parsed_code is None:
            parsed_code = _parse_stock_code(state.get("last_answer") or "")

        if parsed_code is None:
            return 0.0

        return 1.0 if parsed_code == ground_truth else 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Free per-trajectory state."""
        self._instance_dict.pop(instance_id, None)