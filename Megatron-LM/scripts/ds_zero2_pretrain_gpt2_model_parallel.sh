#! /bin/bash

MP_SIZE=8

NUM_WORKERS=4
NUM_GPUS_PER_WORKER=8

NP=32 #Change this later to multiplication of two above.

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_zero2_config.json"
config_json="$script_dir/ds_zero2_config_none.json"

gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 40 \
       --hidden-size 1536 \
       --num-attention-heads 16 \
       --batch-size 32 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .1 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="deepspeed --hostfile=$script_dir/../hostfile --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} $script_dir/../pretrain_gpt2.py $@ ${gpt_options} "
echo ${run_cmd}

eval ${run_cmd}
#echo ${distrun}

set +x
