set -x

PROJECT_DIR="$(pwd)"
ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export VLLM_PROMPT_MAX_IMAGE_PIXELS=602112
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # for vllm0.11.0 with TP
export WANDB_KEY="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"STOCK_AGENT_FAKE_OUTPUT=1

STOCK_AGENT_FAKE_OUTPUT=1
train_path=data/stock_candlestick_2_parallel/train.parquet
test_path=data/stock_candlestick_2_parallel/train.parquet
max_prompt_length=2304
max_response_length=512
max_token_len=$((max_prompt_length + max_response_length))

gen_prompt_bsz=512 # NOTE: no filtering here
filter_groups_metric=acc


ray stop
python3 -m recipe.dapo.main_dapo \
    --config-name="dapo_megatron_trainer" \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=16 \
    data.gen_batch_size=16 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=data/Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_len \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_len \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=4 \
    trainer.use_legacy_worker_impl=disable \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    data.gen_batch_size=8 \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3_vl_30b_megatron' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/stock_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    2>&1 | tee text.txt
