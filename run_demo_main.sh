export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 demo_main.py \
    --model_path "/mnt/data/checkpoints/Wan-AI/Wan2.1-T2V-14B-Diffusers" \
    --transformer_subfolder "distill_4_v2" \
    --sampling_steps 4 \
    --sp_size 8