#export HSAKMT_DEBUG_LEVEL=7
#export AMD_LOG_LEVEL=3
#export NCCL_DEBUG=INFO
deepspeed --num_nodes 1 --num_gpus 8 pretrain_gpt2.py --model-parallel-size=1 --num-layers=50 --hidden-size=4096 --num-attention-heads=32 --seq-length=1024 --batch-size=8 --max-position-embeddings=1024 --train-iters=10 --lr-decay-iters=100 --data-path=/data/DeepSpeed_data/Megatron_wikipedia/my-gpt2_text_document --vocab-file=/data/DeepSpeed_data/Megatron_wikipedia/gpt2-vocab.json --merge-file=/data/DeepSpeed_data/Megatron_wikipedia/gpt2-merges.txt --data-impl=mmap --split=1000,0,0 --distributed-backend=nccl --lr=0.00015 --lr-decay-style=cosine --min-lr=1e-05 --weight-decay=0.01 --clip-grad=1.0 --warmup=0.01 --checkpoint-num-layers=1 --log-interval=1 --deepspeed_config=examples/ds_zero2_10B_cpu.json --contigious-checkpointing --checkpoint-activations --deepspeed --synchronize-each-layer --no-load-optim --checkpoint-in-cpu --fp16 --deepspeed-activation-checkpointing --partition-activations --cpu-optimizer

