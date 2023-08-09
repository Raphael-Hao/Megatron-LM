#!/bin/bash
# This example will start serving the 345M model that is partitioned 8 way tensor parallel
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

torchrun $DISTRIBUTED_ARGS dist_scalability.py \
       --arch attention \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --no-async-tensor-model-parallel-allreduce
