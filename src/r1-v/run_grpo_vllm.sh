cd src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_7b.txt"



CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py --use_vllm True \
    --output_dir /opt/tiger/R1-V/results/ \
    --model_name_or_path /opt/tiger/Qwen2.5-VL-7B-Instruct \
    --dataset_name /opt/tiger/R1-V/data/data \
    --max_prompt_length 512 \
    --max_completion_length 1536 \
    --temperature 0.9 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 400000 \
    --max_steps 13125 \
    --run_name qwen2.5vl-7b-geo-8k \
    --save_steps 1000 \
    --save_only_model true
