#!/bin/sh                                                                                                                                                  

PEAK_LR=0.0001          #学習率
MAX_SENTENCES=64        #1GPUのバッチサイズ、バッチサイズ = MAX_SENTENCES * UPDATE_FREQ * GPU数
UPDATE_FREQ=1           #勾配蓄積
MAX_UPDATE=500000       #学習ステップ数
LOG_INTERVAL=2000        #ログを出力する間隔
SAVE_INTERVAL=10000      #モデルを保存する間隔
WARMUP_UPDATES=10000

mkdir -p results
DATA_DIR=data-bin/japara/

MKL_THREADING_LAYER=GNU \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-train $DATA_DIR --arch roberta_base --task masked_lm --criterion masked_lm \
    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES \
    --max-update $MAX_UPDATE --total-num-update $MAX_UPDATE --update-freq $UPDATE_FREQ \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --no-epoch-checkpoints --seed 88 --log-format simple --log-interval $LOG_INTERVAL --save-interval-updates $SAVE_INTERVAL \
    --tensorboard-logdir ./runs \
    --fp16 --num-workers 0 --fp16-init-scale 4 2>&1 | tee results/train.log