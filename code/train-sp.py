# train.py
import sentencepiece as spm
import os

dir='sp-model'
if not os.path.exists(dir):  # 無ければ
    os.makedirs(dir) 
# 学習
spm.SentencePieceTrainer.Train('--input=data/japara/orig/ja.train, --model_prefix=sp-model/sentencepiece --character_coverage=0.9995 --vocab_size=32000')