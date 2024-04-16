CUDA_VISIBLE_DEVICES=1 python generate.py \
    --load_8bit \
    --base_model './lora-alpaca/mistral-7b-instruct' \
    --lora_weights './lora-alpaca/mistral-7b-instruct-d0_lr2e-4_epoch1_bs16_len512_wp0.05_lora16_8_quan_chat' \
    --data_path '../Data/human_eval/processed/prem_gen.json' \
