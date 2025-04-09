#!/bin/bash
unset http_proxy
unset https_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export WANDB_MODE="online"
export WANDB_API_KEY="9144b562879460494cad9b7abe439e779cfa8af7"
export TOKENIZERS_PARALLELISM=true

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

if [ -z "$MASTER_ADDR" ]; then
  export MASTER_ADDR="localhost"
fi

if [ -z "$MASTER_PORT" ]; then
  export MASTER_PORT=29501
fi

if [ -z "$NODE_RANK" ]; then
  export NODE_RANK=0
fi

if [ -z "$NNODES" ]; then
  export NNODES=1
fi

torchrun \
  --nproc_per_node 8 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --node_rank $NODE_RANK \
  --nnodes $NNODES \
  train_step_distill.py \
  --config scripts/step_distill/14B_16_8_formal.yaml