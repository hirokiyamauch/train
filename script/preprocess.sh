#!/bin/sh                                                                                                                                                  

DIR=data/japara/tokenized

fairseq-preprocess \
    --only-source \
    --trainpref $DIR/ja.train \
    --validpref $DIR/ja.valid \
    --destdir data-bin/japara/ \
    --workers 60