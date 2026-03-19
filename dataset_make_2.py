"""
Preprocess the stock candlestick identification dataset to parquet format for multi-turn RL.

Each sample contains:
  - A user-uploaded screenshot of a stock chart
  - The ground truth: 6-digit stock code extracted from the image filename
    e.g. "Images/001209_20251105_20251223.png" -> "001209"

Usage:
    python make_stock_dataset.py \
        --local_dataset_path /path/to/raw_data.jsonl \
        --images_dir /path/to/Images \
        --local_save_dir /path/to/output \
        --system_prompt_path ./system_prompt.md \
        --train_ratio 0.9
"""

import argparse
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_system_prompt(path: str) -> str:
    """Load system prompt from a markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_ground_truth(image_path: str) -> str | None:
    """
    Extract 6-digit stock code from the image filename.

    Examples:
        "Images/001209_20251105_20251223.png"  -> "001209"
        "Images/600519_20240101_20240301.png"  -> "600519"
    """
    match = re.search(r"(?:^|[/\\])(\d{6})_", image_path)
    if match:
        return match.group(1)
    return None


def extract_user_content(raw_content_parts: list[dict]) -> tuple[str | None, str | None]:
    """
    From the user turn's content list, extract:
      - dynamic_user_text: the portion of the text between the first <time> tag
                           and the trailing marker "用户上传的图片："  (inclusive)
      - image_path:        the image path/url referenced in the message

    Raw content format example:
        [
          {"type": "text",
           "text": "### 身份及任务\\n...<time>2026-03-09</time>\\n<question>...</question>\\n用户上传的图片："},
          {"type": "text",
           "text": "<img_url>Images/001205_20250828_20251021.png</img_url>"},
          {"type": "image_url", "image_url": {"url": "Images/001205_20250828_20251021.png"}}
        ]
    """
    image_path = None
    dynamic_user_text = None

    for part in raw_content_parts:
        ptype = part.get("type", "")
        text = part.get("text", "")

        if ptype == "text":
            # Extract dynamic section: from first <time> up to "用户上传的图片："
            if dynamic_user_text is None:
                match = re.search(r"(<time>.*?用户上传的图片：)", text, re.DOTALL)
                if match:
                    dynamic_user_text = match.group(1).strip()

            # Extract image path from <img_url> tag
            if image_path is None:
                img_match = re.search(r"<img_url>(.*?)</img_url>", text)
                if img_match:
                    image_path = img_match.group(1).strip()

        elif ptype == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url:
                image_path = url  # prefer the explicit image_url entry

    return dynamic_user_text, image_path


def load_raw_dataset(jsonl_path: str) -> list[dict]:
    """Load raw JSONL dataset, one JSON object per line."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: skipping malformed JSON — {e}")
    return samples


def load_image(image_path: str, images_dir: str):
    """Load image as PIL Image object."""
    candidates = [
        image_path,
        os.path.join(images_dir, os.path.basename(image_path)),
        os.path.join(images_dir, image_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return Image.open(p).copy()  # .copy() to avoid lazy-load issues after file close
    print(f"[WARN] Image not found: {image_path}")
    return None


# ---------------------------------------------------------------------------
# Core dataset builder
# ---------------------------------------------------------------------------

def build_hf_dataset(
    raw_samples: list[dict],
    system_prompt: str,
    images_dir: str,
) -> datasets.Dataset:
    """
    Convert raw JSONL samples into a HuggingFace Dataset with the schema
    expected by the multi-turn RL tool-agent pipeline.
    """
    records = []
    skipped = 0

    for idx, sample in enumerate(raw_samples):
        messages = sample.get("messages", [])
        if not messages:
            print(f"[WARN] Sample {idx}: no messages field, skipping.")
            skipped += 1
            continue

        user_turn = messages[1]
        if user_turn.get("role") != "user":
            print(f"[WARN] Sample {idx}: first message is not a user turn, skipping.")
            skipped += 1
            continue

        user_content_parts = user_turn.get("content", [])
        # dynamic_user_text, image_path = extract_user_content(user_content_parts)
        dynamic_user_text = user_content_parts[1]['text']
        image_path = user_content_parts[0]['image_url']['url']

        if image_path is None:
            print(f"[WARN] Sample {idx}: no image path found, skipping.")
            skipped += 1
            continue

        # ground_truth = extract_ground_truth(image_path)
        ground_truth = sample['choices'][0]['message']['content'][0]['text']
        if ground_truth is None:
            print(f"[WARN] Sample {idx}: could not extract ground truth from '{image_path}', skipping.")
            skipped += 1
            continue

        # ------------------------------------------------------------------
        # Build user message text
        # The dynamic portion already contains <time> and <question> blocks.
        # Fall back to a plain question if regex extraction failed.
        # ------------------------------------------------------------------
        if dynamic_user_text:
            user_message_text = dynamic_user_text + "<image>"  # append here
        else:
            user_message_text = "请你结合这张交易软件的截图，确定股票代码。<image>"
            for part in user_content_parts:
                if part.get("type") == "text":
                    q_match = re.search(r"<question>(.*?)</question>", part["text"], re.DOTALL)
                    if q_match:
                        user_message_text = q_match.group(1).strip() + "<image>"
                        break

        # ------------------------------------------------------------------
        # Multi-modal user content: text + image reference
        # ------------------------------------------------------------------
        user_content = [
            {"type": "text", "text": user_message_text},
            {"type": "image_url", "image_url": {"url": image_path}},
        ]

        # Load image bytes so the dataset is self-contained
        image_bytes = load_image(image_path, images_dir)

        # ------------------------------------------------------------------
        # Final record — mirrors the gsm8k / geo3k schema
        # ------------------------------------------------------------------
        record = {
            "data_source": "hithink_stock_candlestick",
            # Tells the rollout server which AgentLoop subclass to use.
            # We will register "stock_chart_agent" in Task 3.
            # "agent_name": "stock_chart_agent",
            "agent_name": "stock_chart_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message_text,
                },
            ],
            # Stored as a list to be consistent with geo3k multimodal format
            "images": [image_bytes] if image_bytes is not None else [],
            "image_path": image_path,
            "ability": "stock_identification",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,   # 6-digit string, e.g. "001209"
            },
            "extra_info": {
                "index": idx,
                "image_path": image_path,
                "ground_truth": ground_truth,
                # Signals to the rollout runner that tool kwargs are needed
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    # The reward tool receives the ground truth at creation time
                    # so it can score the model's <FINISHED> answer at execute time.
                    "calc_stock_reward": {
                        "create_kwargs": {"ground_truth": ground_truth},
                    },
                },
                "interaction_kwargs": {
                    "ground_truth": ground_truth,
                    "image_path": image_path,
                },
            },
        }
        # if not record["pr"]:
        #     print(f"[WARN] Sample {idx}: images list is empty, skipping.")
        #     skipped += 1
        #     continue

        records.append(record)
    # print(records[0])
    print(f"[INFO] Built {len(records)} valid records ({skipped} skipped) from {len(raw_samples)} raw samples.")
    return datasets.Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build multi-turn RL parquet dataset for stock chart identification."
    )
    parser.add_argument(
        "--local_dataset_path",
        required=True,
        help="Path to the raw JSONL dataset file.",
    )
    parser.add_argument(
        "--images_dir",
        default="./Images",
        help="Root directory containing the stock chart images referenced by the JSONL.",
    )
    parser.add_argument(
        "--system_prompt_path",
        default="./system_prompt.md",
        help="Path to the system prompt markdown file.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/stock_candlestick_multiturn",
        help="Output directory for the parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS destination to copy outputs to.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of samples used for training (remainder becomes test set).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load system prompt
    # ------------------------------------------------------------------
    system_prompt = load_system_prompt(args.system_prompt_path)
    print(f"[INFO] Loaded system prompt ({len(system_prompt)} chars) from '{args.system_prompt_path}'")

    # ------------------------------------------------------------------
    # Load raw JSONL
    # ------------------------------------------------------------------
    raw_samples = load_raw_dataset(args.local_dataset_path)
    print(f"[INFO] Loaded {len(raw_samples)} raw samples from '{args.local_dataset_path}'")

    # ------------------------------------------------------------------
    # Build HuggingFace dataset
    # ------------------------------------------------------------------
    full_dataset = build_hf_dataset(raw_samples, system_prompt, args.images_dir)

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    split = full_dataset.train_test_split(test_size=round(1.0 - args.train_ratio, 6), seed=args.seed)
    train_dataset = split["train"]
    test_dataset = split["test"]
    print(f"[INFO] Split -> Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # Save to parquet
    # ------------------------------------------------------------------
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"[INFO] Saved -> {train_path}")
    print(f"[INFO] Saved -> {test_path}")

    # ------------------------------------------------------------------
    # Optional HDFS upload
    # ------------------------------------------------------------------
    
    import json
    def make_json_serializable(obj):
        """Recursively replace non-JSON-serializable objects (e.g. PIL images) with 'image'."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return "image"

    def save_jsonl(dataset, path):
        with open(path, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(make_json_serializable(example), ensure_ascii=False) + "\n")

    save_jsonl(train_dataset, os.path.join(local_save_dir, "train.jsonl"))
    save_jsonl(test_dataset, os.path.join(local_save_dir, "test.jsonl"))
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"[INFO] Copied to HDFS: {args.hdfs_dir}")