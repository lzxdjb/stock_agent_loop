"""
quick_test_system_prompt.py

Load a parquet dataset and quickly test a system prompt using vLLM.
Usage:
    python quick_test_system_prompt.py \
        --parquet_path ~/data/stock_candlestick_multiturn/test.parquet \
        --system_prompt_path ./system_prompt.md \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --num_samples 10
"""

import argparse
import os
import re
from io import BytesIO

import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_parquet(path: str) -> pd.DataFrame:
    path = os.path.expanduser(path)
    df = pd.read_parquet(path)
    print(f"[INFO] Loaded {len(df)} rows from '{path}'")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def extract_pil_image(images_field) -> Image.Image | None:
    """
    Handle multiple possible storage formats for the 'images' column:
      - list of PIL.Image
      - list of dicts with 'bytes' key  (HF Image feature serialized)
      - list of raw bytes
    """
    if not images_field:
        return None

    img = images_field[0]

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, dict):
        raw = img.get("bytes") or img.get("path")
        if isinstance(raw, bytes):
            return Image.open(BytesIO(raw)).convert("RGB")

    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")

    print(f"[WARN] Unknown image type: {type(img)}")
    return None

import base64

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompt_for_vllm(user_text: str, system_prompt: str, pil_image: Image.Image) -> list[dict]:
    """
    Embed PIL image directly in the message content list.
    This is the correct API for vLLM >= 0.4.x
    """
    clean_text = user_text.replace("<image>", "").strip()

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_base64(pil_image)},  # must be string
                },
                {"type": "text", "text": clean_text},
            ],
        },
    ]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quick system-prompt tester using vLLM + parquet dataset.")
    parser.add_argument("--parquet_path", required=True, help="Path to test.parquet or train.parquet")
    parser.add_argument("--system_prompt_path", default="./system_prompt.md", help="Path to system prompt .md file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="vLLM model name or local path")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load system prompt & dataset
    # ------------------------------------------------------------------
    system_prompt = load_system_prompt(args.system_prompt_path)
    print(f"[INFO] System prompt ({len(system_prompt)} chars):\n{system_prompt[:200]}...\n")

    df = load_parquet(args.parquet_path)
    samples = df.head(args.num_samples)

    # ------------------------------------------------------------------
    # Build vLLM engine
    # ------------------------------------------------------------------
    print(f"[INFO] Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Run inference row by row
    # ------------------------------------------------------------------
    results = []
    for i, (_, row) in enumerate(samples.iterrows()):
        ground_truth = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) \
            else row.get("ground_truth", "???")

        # Reconstruct user text from prompt field
        prompt_messages = row["prompt"]
        user_text = ""
        for msg in prompt_messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_text = content if isinstance(content, str) else str(content)
                break

        # Extract PIL image
        pil_image = extract_pil_image(row.get("images"))
        if pil_image is None:
            print(f"[WARN] Sample {i}: no image, skipping.")
            continue

        # Build chat messages
        messages = build_prompt_for_vllm(user_text, system_prompt, pil_image)

        # vLLM multi-modal inference
        output = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            # multi_modal_data={"image": pil_image},
        )

        response_text = output[0].outputs[0].text.strip()

        # Simple accuracy check: does the response contain the ground truth code?
        hit = ground_truth in response_text

        results.append({
            "index": i,
            "ground_truth": ground_truth,
            "response": response_text,
            "hit": hit,
        })

        print(f"\n{'='*60}")
        print(f"[{i}] GT: {ground_truth} | Hit: {hit}")
        print(f"Response: {response_text}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if results:
        accuracy = sum(r["hit"] for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f"[SUMMARY] Tested {len(results)} samples")
        print(f"[SUMMARY] Accuracy (GT in response): {accuracy:.1%}")


if __name__ == "__main__":
    main()