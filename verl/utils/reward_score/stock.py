"""
verl/trainer/ppo/reward/stock.py

Reward scoring for the stock candlestick identification task.

Plugs into default_compute_score via data_source == "hithink_stock_candlestick".

Score breakdown
---------------
  acc_reward    (weight: 1 - format_score)
      1.0  — the extracted 6-digit code matches ground truth exactly
      0.0  — a valid 6-digit code was found but it is wrong
     -0.1  — no parseable 6-digit code found at all

  format_reward (weight: format_score)
      1.0  — response contains a <FINISHED> block followed by a 6-digit code
             in the expected "股票代码：XXXXXX" format
      0.0  — format requirements not met

Default format_score = 0.1 (same weight as geo3k), so a fully correct +
well-formatted answer scores 1.0 and a correct-but-poorly-formatted answer
scores 0.9.
"""

import re

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Strict: <FINISHED> ... 股票代码候选：XXXXXX, YYYYYY, ...
_FORMAT_STRICT_RE = re.compile(
    r"<FINISHED>.*?股票代码候选[：:]\s*([\d,，\s]+)",
    re.DOTALL | re.IGNORECASE,
)

# Loose fallback: <FINISHED> followed by any 6-digit numbers
_FORMAT_LOOSE_RE = re.compile(
    r"<FINISHED>.*?(\d{6}(?:[,，\s]+\d{6})*)",
    re.DOTALL,
)

_SIX_DIGIT_RE = re.compile(r"\b(\d{6})\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_answers(predict_str: str) -> list[str]:
    """
    Extract all candidate 6-digit stock codes from the model's response.
    Returns an ordered list (highest confidence first), empty list if none found.
    """
    m = _FORMAT_STRICT_RE.search(predict_str)
    if m:
        return _SIX_DIGIT_RE.findall(m.group(1))

    m = _FORMAT_LOOSE_RE.search(predict_str)
    if m:
        return _SIX_DIGIT_RE.findall(m.group(1))

    # Last-resort: all bare 6-digit tokens in the full string
    matches = _SIX_DIGIT_RE.findall(predict_str)
    return matches[-5:] if matches else []   # take last 5 at most





def format_reward(predict_str: str) -> float:
    """1.0 if <FINISHED> is followed by at least one 6-digit code."""
    return 1.0 if _FORMAT_LOOSE_RE.search(predict_str) else 0.0



def acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    +1.0  ground truth is anywhere in the candidate list
     0.0  candidates found but none match
    -0.1  no parseable 6-digit code at all
    """
    candidates = extract_answers(predict_str)
    if not candidates:
        return -0.1
    return 1.0 if str(ground_truth).strip() in candidates else 0.0

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_score(
    predict_str: str,
    ground_truth: str,
    format_score: float = 0.1,
) -> float:
    """
    Weighted combination of accuracy and format rewards.

    Parameters
    ----------
    predict_str  : full model response string
    ground_truth : 6-digit stock code string, e.g. "001209"
    format_score : weight given to the format reward (default 0.1)

    Returns
    -------
    float in roughly [-0.1, 1.0]
    """
    acc = acc_reward(predict_str, ground_truth)
    fmt = format_reward(predict_str)
    return (1.0 - format_score) * acc + format_score * fmt