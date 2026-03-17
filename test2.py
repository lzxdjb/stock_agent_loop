Below is the code I write for making the RL dataset

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


def load_image_bytes(image_path: str, images_dir: str) -> bytes | None:
    """
    Try to load raw image bytes.  Tries several path resolution strategies:
      1. image_path as-is (absolute or relative to cwd)
      2. basename of image_path under images_dir
      3. image_path joined under images_dir
    """
    candidates = [
        image_path,
        os.path.join(images_dir, os.path.basename(image_path)),
        os.path.join(images_dir, image_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
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

        user_turn = messages[0]
        if user_turn.get("role") != "user":
            print(f"[WARN] Sample {idx}: first message is not a user turn, skipping.")
            skipped += 1
            continue

        user_content_parts = user_turn.get("content", [])
        dynamic_user_text, image_path = extract_user_content(user_content_parts)

        if image_path is None:
            print(f"[WARN] Sample {idx}: no image path found, skipping.")
            skipped += 1
            continue

        ground_truth = extract_ground_truth(image_path)
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
            user_message_text = dynamic_user_text
        else:
            # Try to pull just the <question> block as a minimal fallback
            user_message_text = "请你结合这张交易软件的截图，确定股票代码。"
            for part in user_content_parts:
                if part.get("type") == "text":
                    q_match = re.search(r"<question>(.*?)</question>", part["text"], re.DOTALL)
                    if q_match:
                        user_message_text = q_match.group(1).strip()
                        break

        # ------------------------------------------------------------------
        # Multi-modal user content: text + image reference
        # ------------------------------------------------------------------
        user_content = [
            {"type": "text", "text": user_message_text},
            {"type": "image_url", "image_url": {"url": image_path}},
        ]

        # Load image bytes so the dataset is self-contained
        image_bytes = load_image_bytes(image_path, images_dir)

        # ------------------------------------------------------------------
        # Final record — mirrors the gsm8k / geo3k schema
        # ------------------------------------------------------------------
        record = {
            "data_source": "hithink_stock_candlestick",
            # Tells the rollout server which AgentLoop subclass to use.
            # We will register "stock_chart_agent" in Task 3.
            "agent_name": "stock_chart_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
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
        records.append(record)
    print(record[0])
    print(f"[INFO] Built {len(records)} valid records ({skipped} skipped) from {len(raw_samples)} raw samples.")
    return datasets.Dataset.from_list(records)


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
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"[INFO] Copied to HDFS: {args.hdfs_dir}")
        
    
Now it will output the error in 
    return datasets.Dataset.from_list(records)


[INFO] Loaded system prompt (1856 chars) from './system_prompt.md'
[INFO] Loaded 1036 raw samples from './dataset1/paas_1036.jsonl'
[INFO] Built 1036 valid records (0 skipped) from 1036 raw samples.
Traceback (most recent call last):
  File "/usr/lib/python3.12/runpy.py", line 198, in _run_module_as_main
    return _run_code(code, main_globals, None,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/runpy.py", line 88, in _run_code
    exec(code, run_globals)
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py", line 71, in <module>
    cli.main()
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 508, in main
    run()
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 358, in run_file
    runpy.run_path(target, run_name="__main__")
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 310, in run_path
    return _run_module_code(code, init_globals, run_name, pkg_name=pkg_name, script_name=fname)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 127, in _run_module_code
    _run_code(code, mod_globals, init_globals, mod_name, mod_spec, pkg_name, script_name)
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 118, in _run_code
    exec(code, run_globals)
  File "/mnt/data/HithinkOmniSSD/user_workspace/leizhengxing/verl/dataset_make.py", line 311, in <module>
    full_dataset = build_hf_dataset(raw_samples, system_prompt, args.images_dir)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/HithinkOmniSSD/user_workspace/leizhengxing/verl/dataset_make.py", line 250, in build_hf_dataset
    return datasets.Dataset.from_list(records)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/datasets/arrow_dataset.py", line 1068, in from_list
    return cls.from_dict(mapping, features, info, split)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/datasets/arrow_dataset.py", line 1022, in from_dict
    pa_table = InMemoryTable.from_pydict(mapping=mapping)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/datasets/table.py", line 757, in from_pydict
    return cls(pa.Table.from_pydict(*args, **kwargs))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/table.pxi", line 1985, in pyarrow.lib._Tabular.from_pydict
  File "pyarrow/table.pxi", line 6401, in pyarrow.lib._from_pydict
  File "pyarrow/array.pxi", line 405, in pyarrow.lib.asarray
  File "pyarrow/array.pxi", line 256, in pyarrow.lib.array
  File "pyarrow/array.pxi", line 118, in pyarrow.lib._handle_arrow_array_protocol
  File "/usr/local/lib/python3.12/dist-packages/datasets/arrow_writer.py", line 307, in __arrow_array__
    out = pa.array(cast_to_python_objects(data, only_1d_for_numpy=True))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/array.pxi", line 375, in pyarrow.lib.array
  File "pyarrow/array.pxi", line 46, in pyarrow.lib._sequence_to_array
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values

First it seems like no row is skippied every row in the records has the same format

I think the error is in the images
For example when I check the 


records[0]['images']


type(records[0]['images'])
<class 'list'>

type(records[0]['images'][0])
<class 'bytes'>

Below is the code which can work:
    

# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 Reallm Labs Ltd. or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/geo3k_multiturn_w_tool",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "hiyouga/geometry3k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            # print("problem: ", problem, "problem_type: ", type(problem))
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")
            print("images: ", images, "images: ", type(images))
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should use the `calc_geo3k_reward` tool after step by step solving the question, "
                            "before generate final answer at least once and refine your answer if necessary. "
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "calc_geo3k_reward": {
                            "create_kwargs": {"ground_truth": answer},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

I also check the image types in above code:

images:  [<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=410x265 at 0x7FAD081CCE00>] images:  <class 'list'>

So may be we can mimic this image format?