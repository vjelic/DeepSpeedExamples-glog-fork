#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=lamb_32k_seq512_output
OUTPUT_DIR=${base_dir}/bert_model_outputs

# Assumes job name in previous seq128 run, will resume training from epoch 150

mkdir -p $OUTPUT_DIR

deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large_lamb_pipeclean.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--print_steps 100 \
--deepspeed \
--deepspeed_transformer_kernel \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz32k_lamb_config_seq512_pipeclean.json \
--data_path_prefix /data/bert \
--rewarmup \
--lr_schedule "EE" \
--attention_dropout_checkpoint \
--lr_offset 0.0 \
&> ${JOB_NAME}.log
