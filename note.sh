export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
python -m examples.data_preprocess.geo3k_multiturn_w_tool


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
python -m examples.data_preprocess.gsm8k_tool_agent_loop_with_guard --local_save_dir ./data/gsm8k_tool_agent_loop_with_guard


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
python -m examples.data_preprocess.gsm8k_tool_agent_loop_with_assistant --local_save_dir ./data/gsm8k_tool_agent_loop_with_assistant


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
python -m examples.data_preprocess.gsm8k_tool_agent_loop --local_save_dir ./data/gsm8k_tool_agent_loop



export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
python -m examples.data_preprocess.geo3k_multiturn_w_tool --local_save_dir ./data/geo3k_multiturn_w_tool

python -m examples.data_preprocess.gsm8k --local_save_dir ./data/gsm8k


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
python -m examples.data_preprocess.geo3k --local_save_dir ./data/geo3k



huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct --revision main --cache-dir /mnt/data/HithinkOmni/user_workspace/leizhengxing/verl/data/Qwen3-VL-30B-A3B-Instruct


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir ./data/Qwen3-VL-30B-A3B-Instruct


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen3-4B --local-dir ./data/Qwen3-4B



export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./data/Qwen2.5-VL-7B-Instruct



export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./data/Qwen/Qwen3.5-4B





huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Thinking --local-dir ./data/Qwen3-VL-30B-A3B-Thinking


huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./data/Qwen2.5-VL-3B-Instruct


huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ./data/Qwen3-Omni-30B-A3B-Instruct



export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir ./data/Qwen3-VL-30B-A3B-Instruct



export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct  --local-dir ./data/Qwen3-VL-4B-Instruct


#######

export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /mnt/data/HithinkOmni/user_workspace/leizhengxing/verl/data/Qwen2.5-VL-3B-Instruct 




export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir .



export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct  --local-dir ./data/Qwen3-VL-8B-Instruct


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen3-0.6B  --local-dir ./data/Qwen3-0.6B


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct  --local-dir ./data/Qwen/Qwen3-VL-2B-Instruct





python dataset_make.py \
  --local_dataset_path ./dataset1/paas_1036.jsonl \
  --images_dir ./dataset1/Images/ \
  --system_prompt_path ./system_prompt.md \
  --local_save_dir ./data/stock_candlestick


python dataset_make_2.py \
  --local_dataset_path ./data/dataset/processed_merge_30-45.jsonl \
  --images_dir ./data/dataset/ \
  --system_prompt_path ./system_prompt.md \
  --local_save_dir ./data/stock_candlestick_2


python dataset_make_2_parallel.py \
  --local_dataset_path ./data/dataset/processed_merge_30-45.jsonl \
  --images_dir ./data/dataset/ \
  --system_prompt_path ./system_prompt.md \
  --local_save_dir ./data/stock_candlestick_2_parallel



python vllm_quick_test.py \
    --parquet_path data/stock_candlestick/train.parquet \
    --system_prompt_path ./system_prompt.md \
    --model data/Qwen3-VL-8B-Instruct \
    --num_samples 20 \
    --temperature 0.0

git config --global user.email "zhengxinglei539@gmail.com"
git config --global user.name "lzxdjb"
  


export http_proxy="http://hexin:hx300033@10.244.57.246:30100"
export https_proxy="http://hexin:hx300033@10.244.57.246:30100"




git remote add gitlab https://git-cc.myhexin.com:6443/leizhengxing/stock-agent-rl.git
git push -u gitlab main