export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=4 infer.py \
    --model_path "/mnt/workspace/checkpoints/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/" \
    --sp_size 4 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --sampling_steps 32 \
    --cfg 5.0 \
    --seed 2002 \
    --prompt_file scripts/prompt_t2v.txt \
    --flow_shift 7.0 \
    --transformer_subfolder "transformer" \
    --output_dir "output/1.3B_32"