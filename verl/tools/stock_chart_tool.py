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


def _parse_stock_codes(raw: str) -> list[str]:
    """
    Extract all 6-digit stock codes from the model's answer string.
    Handles: "001209, 300033", "股票代码候选：001209、300033", "#### 001209", etc.
    Returns ordered list, empty if nothing found.
    """
    # Strip market suffixes like .SH / .SZ wherever they appear
    raw = re.sub(r"\.(?:SH|SZ|BJ|sh|sz|bj)\b", "", raw)
    return re.findall(r"\b(\d{6})\b", raw)



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
            "last_answers": [],      # was: "last_answer": None
            "best_reward":  0.0,
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

        candidates = _parse_stock_codes(raw_answer)

        if not candidates:
            feedback = (
                f"无法从您的回答 '{raw_answer}' 中解析出有效的6位股票代码。"
                f"请确保提交纯数字的6位股票代码，例如 '001209' 或 '001209, 300033'。"
            )
            state["last_answers"] = []
            return ToolResponse(text=feedback), -0.1, {}

        ground_truth = state["ground_truth"]
        is_correct = ground_truth in candidates
        reward = 1.0 if is_correct else 0.0

        if reward > state["best_reward"]:
            tool_reward = reward - state["best_reward"]
            state["best_reward"] = reward
        else:
            tool_reward = -0.05

        state["last_answers"] = candidates

        if is_correct:
            feedback = (
                f"✓ 正确！目标股票代码 {ground_truth} 在您的候选列表 {candidates} 中。"
            )
        else:
            feedback = (
                f"✗ 错误。您的候选列表为 {candidates}，目标股票代码为 {ground_truth}。"
                f"请重新分析后再试。"
            )

        return ToolResponse(text=feedback), tool_reward, {}

    async def calc_reward(self, instance_id: str, parsed_code: str | None = None, **kwargs) -> float:
        state = self._instance_dict[instance_id]
        ground_truth = state["ground_truth"]
        if parsed_code is not None:
            return 1.0 if parsed_code == ground_truth else 0.0
        # Multi-candidate path
        candidates = state.get("last_answers") or []
        return 1.0 if ground_truth in candidates else 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Free per-trajectory state."""
        self._instance_dict.pop(instance_id, None)