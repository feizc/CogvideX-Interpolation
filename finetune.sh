

accelerate launch --mixed_precision="bf16" finetune.py \
    --data_path=video600w.json \
    --pretrained_model_name_or_path=/maindata/data/shared/public/multimodal/share/zhengcong.fei/ckpts/CogVideoX-5b-I2V \
    --mixed_precision="bf16" \
    --checkpointing_steps 200