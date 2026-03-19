import argparse
import json
import os
from multiprocessing import Pool, cpu_count

import datasets
from PIL import Image


# ---------------------------------------------------------------------------
# Fast image loader (NO PIL)
# ---------------------------------------------------------------------------
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
# Process ONE sample (for multiprocessing)
# ---------------------------------------------------------------------------
def process_sample(args):
    idx, sample, system_prompt, images_dir = args

    try:
        messages = sample.get("messages", [])
        if len(messages) < 2:
            return None

        user_turn = messages[1]
        if user_turn.get("role") != "user":
            return None

        user_content_parts = user_turn.get("content", [])

        # Fast extraction (NO regex)
        dynamic_user_text = user_content_parts[1]["text"]
        image_path = user_content_parts[0]["image_url"]["url"]

        ground_truth = sample["choices"][0]["message"]["content"][0]["text"]

        user_message_text = dynamic_user_text + "<image>"

        image_bytes = load_image(image_path, images_dir)

        return {
            "data_source": "hithink_stock_candlestick",
            "agent_name": "stock_chart_agent",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_text},
            ],
            "images": [image_bytes] if image_bytes is not None else [],
            "image_path": image_path,
            "ability": "stock_identification",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "index": idx,
                "image_path": image_path,
                "ground_truth": ground_truth,
                "need_tools_kwargs": True,
                "tools_kwargs": {
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

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parallel dataset builder
# ---------------------------------------------------------------------------
def build_dataset_parallel(raw_samples, system_prompt, images_dir, num_workers):
    args = [(i, s, system_prompt, images_dir) for i, s in enumerate(raw_samples)]

    with Pool(num_workers) as pool:
        results = list(pool.imap_unordered(process_sample, args, chunksize=32))

    records = [r for r in results if r is not None]

    print(f"[INFO] Built {len(records)} records from {len(raw_samples)} samples")

    return datasets.Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Optional JSONL saver (FAST)
# ---------------------------------------------------------------------------
def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            row["images"] = ["<image_bytes>"] if row["images"] else []
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_dataset_path", required=True)
    parser.add_argument("--images_dir", default="./Images")
    parser.add_argument("--system_prompt_path", default="./system_prompt.md")
    parser.add_argument("--local_save_dir", default="./output")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    # NEW
    parser.add_argument("--num_workers", type=int, default=min(cpu_count(), 16))
    parser.add_argument("--save_jsonl", action="store_true")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load system prompt
    # ------------------------------------------------------------------
    with open(args.system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    print(f"[INFO] Loaded system prompt")

    # ------------------------------------------------------------------
    # Load raw dataset
    # ------------------------------------------------------------------
    raw_samples = load_jsonl(args.local_dataset_path)
    print(f"[INFO] Loaded {len(raw_samples)} samples")

    # ------------------------------------------------------------------
    # Build dataset (PARALLEL 🚀)
    # ------------------------------------------------------------------
    dataset = build_dataset_parallel(
        raw_samples,
        system_prompt,
        args.images_dir,
        args.num_workers,
    )

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    split = dataset.train_test_split(
        test_size=round(1.0 - args.train_ratio, 6),
        seed=args.seed,
    )

    train_dataset = split["train"]
    test_dataset = split["test"]

    print(f"[INFO] Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # Save parquet (FAST)
    # ------------------------------------------------------------------
    os.makedirs(args.local_save_dir, exist_ok=True)

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    test_path = os.path.join(args.local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path, compression="snappy")
    test_dataset.to_parquet(test_path, compression="snappy")

    print(f"[INFO] Saved parquet files")

    # ------------------------------------------------------------------
    # Optional JSONL
    # ------------------------------------------------------------------
    save_jsonl(train_dataset, os.path.join(args.local_save_dir, "train.jsonl"))
    save_jsonl(test_dataset, os.path.join(args.local_save_dir, "test.jsonl"))
    print(f"[INFO] Saved JSONL files")