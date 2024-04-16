CUDA_VISIBLE_DEVICES=0 python finetune-instruct.py \
    --base model './lora-alpaca/mistral-7b-instruct' \
    --data_path '../Data/general/training_data_d0.json' \
    --dev_data_path '../Data/general/training_data_5000.json' \
    --val_set_size 0 \
    --output_dir './lora-alpaca/mistral-7b-instruct-d0_lr2e-4_epoch1_bs16_len512_wp0.05_lora16_8_quan_chat' \
    --num_epochs 1 \
    --learning_rate 2e-4 \
    --micro_batch_size 16 \
    --cutoff_len 512 \
    --group_by_length \
    --wandb_project 'rule-verification' \
    --wandb_run_name 'mistral-7b-instruct-d0_lr2e-4_epoch1_bs16_len512_wp0.05_lora16_8_quan_chat' \
    --lora_r 16 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head]'