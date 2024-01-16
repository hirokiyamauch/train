#!/bin/sh                                                                                                                                                  

PEAK_LR=0.0005          #学習率
MAX_SENTENCES=16        #1GPUのバッチサイズ、バッチサイズ = MAX_SENTENCES * UPDATE_FREQ * GPU数
UPDATE_FREQ=8           #勾配蓄積
TOTAL_UPDATES=500000       #学習ステップ数
LOG_INTERVAL=2000       #ログを出力する間隔
SAVE_INTERVAL=100000      #モデルを保存する間隔
WARMUP_UPDATES=10000

mkdir -p results
DATA_DIR=data-bin/japara/

MKL_THREADING_LAYER=GNU \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-train $DATA_DIR --arch bart_base --task denoising --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --mask-length span-poisson --mask 0.35 --poisson-lambda 3.5 --permute-sentences 1 --replace-length 1 \
    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES  --rotate 0.0 \
    --max-update $TOTAL_UPDATES --total-num-update $TOTAL_UPDATES --update-freq $UPDATE_FREQ \
    --tensorboard-logdir ./runs \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --no-epoch-checkpoints --seed 88 --log-format simple --log-interval $LOG_INTERVAL --save-interval-updates $SAVE_INTERVAL \
    --fp16 --num-workers 0 --fp16-init-scale 4 2>&1 | tee results/train.log                                                                                                                                                