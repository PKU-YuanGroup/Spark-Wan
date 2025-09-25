export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 infer.py \
    --model_path "/mnt/data/checkpoints/Wan-AI/Wan2.1-T2V-14B-Diffusers" \
    --sp_size 4 \
    --height 720 \
    --width 1280     \
    --num_frames 81 \
    --sampling_steps 4 \
    --cfg 0.0 \
    --seed 2002 \
    --prompt_file scripts/prompt_t2v.txt \
    --flow_shift 7.0 \
    --lora_path "/mnt/data/lzj/codes/spark-wan/D14B_8_4_formal_psdual_ema/checkpoint-300/ema_model.safetensors" \
    --transformer_subfolder "transformer" \
    --output_dir "output/formal_14_4_2"