#!/bin/bash
dataset=../data/AMR/amr_2.0
NGPU=3
LR=1e-3
RNN_SZ=256
CKPT="ckpt_$(date +%m-%d-%H-%M)"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--ngpu) NGPU=$2; shift ;;
        -l|--lr) LR=$2; shift ;;
        -r|--rnn_sz) RNN_SZ=$2; shift ;;
        -c|--ckpt) CKPT=$2; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
python3 train.py --token_vocab ${dataset}/token_vocab\
    --concept_vocab ${dataset}/concept_vocab\
    --predictable_token_vocab ${dataset}/predictable_token_vocab\
    --relation_vocab ${dataset}/relation_vocab\
    --token_char_vocab ${dataset}/token_char_vocab\
    --concept_char_vocab ${dataset}/concept_char_vocab\
    --train_data ${dataset}/train.txt.features.preproc.json\
    --dev_data ${dataset}/dev.txt.features.preproc.json\
    --rnn_hidden_size $RNN_SZ\
    --token_char_dim 32\
    --token_dim 300\
    --concept_char_dim 32\
    --concept_dim 300\
    --rel_dim 100\
    --cnn_filter 3 256\
    --char2word_dim 128\
    --char2concept_dim 128\
    --embed_dim 512\
    --ff_embed_dim 1024\
    --num_heads 8\
    --snt_layers 1\
    --graph_layers 4\
    --inference_layers 3\
    --dropout 0.2\
    --unk_rate 0.33\
    --epochs 1000\
    --train_batch_size 44444\
    --dev_batch_size 22222\
    --lr $LR\
    --warmup_steps 2000\
    --print_every 100\
    --eval_every 1000\
    --ckpt $CKPT\
    --world_size $NGPU\
    --gpus $NGPU\
    --MASTER_ADDR localhost\
    --MASTER_PORT 29505\
    --start_rank 0
