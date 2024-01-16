#!/bin/sh

LOAD_DIR=data/orig
SAVE_DIR=data/japara

N_valid=3000
N=`wc -l ${LOAD_DIR}/small_ja.txt | awk '{print $1}'`
N_train=$(($N-$N_valid))


mkdir -p $SAVE_DIR
mkdir -p $SAVE_DIR/orig
mkdir -p $SAVE_DIR/tokenized 

echo create $N_train ja.train
echo create $N_valid ja.valid

head -n $N_train ${LOAD_DIR}/small_ja.txt > $SAVE_DIR/orig/ja.train
tail -n $N_valid ${LOAD_DIR}/small_ja.txt  > $SAVE_DIR/orig/ja.valid