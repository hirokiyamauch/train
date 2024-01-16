import argparse
import collections
import logging

import numpy as np
import rootutils
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

# from model import MLP
from models.model_sev import MLPLightningModule as MLP_sev

# from scipy.stats import pearsonr
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

# import os
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="None")
parser.add_argument("--batch_size", default=256)
parser.add_argument("--cuda", default="cuda:0")
# MODEL_PATH
parser.add_argument("--tokenizer", default="sentence-transformers/LaBSE")
parser.add_argument("--model", default="sentence-transformers/LaBSE")

args = parser.parse_args()
device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

def embedding(
    tokenizer: PreTrainedTokenizer,
    base_model: PreTrainedModel,
    model: nn.Module,
    sentences: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    print("embedding")
    base_model.to(device)
    model.to(device)

    all_embeddings = []

    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    # print(sentences_sorted[10], length_sorted_idx[:10])

    for i in range(-(-len(sentences) // batch_size)):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]

        encoded = tokenizer(
            sentence_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = base_model(**encoded.to(device))

        # last hidden state
        embeddings = outputs[0][:, 0, :]

        embeddings = model(embeddings)[0]

        all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_embeddings = torch.stack(all_embeddings)

    return all_embeddings


def split_emb(sentence_batch,tokenizer,  base_model):
        encoded = tokenizer(sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = base_model(**encoded.to(device))

        # last hidden state
        embeddings = outputs[0][:, 0, :]
        
        return embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cos_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def cos_emb(src_emb, trg_emb):
    for s, t in zip(src_emb, trg_emb):
        print(cos_similarity(s, t)) 
        
def embedding_pare(tokenizer, labse_model, base_model, src_sentences, trg_sentences, batch_size, device):
    print("embeding_pare")
    base_model.to(device)
    labse_model.to(device)

    src_embeddings = []
    trg_embeddings = []

    s_length_sorted_idx = np.argsort([-len(sen) for sen in src_sentences])
    src_sentences_sorted = [src_sentences[idx] for idx in s_length_sorted_idx]
    t_length_sorted_idx = np.argsort([-len(sen) for sen in trg_sentences])
    trg_sentences_sorted = [trg_sentences[idx] for idx in t_length_sorted_idx]

    for i in tqdm(range(-(-len(src_sentences) // batch_size))):
        src_sentence_batch = src_sentences_sorted[i * batch_size : (i + 1) * batch_size]
        trg_sentence_batch = trg_sentences_sorted[i * batch_size : (i + 1) * batch_size]

        s_encoded = tokenizer(src_sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        t_encoded = tokenizer(trg_sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            s_out = labse_model(**s_encoded.to(device))
            t_out = labse_model(**t_encoded.to(device))
            
            s_embeddings = mean_pooling(s_out, s_encoded['attention_mask'])
            t_embeddings = mean_pooling(t_out, t_encoded['attention_mask'])
            s_embeddings = F.normalize(s_embeddings, p=2, dim=1)
            t_embeddings = F.normalize(t_embeddings, p=2, dim=1)
                
            s_out, t_out = base_model(s_embeddings.to(device), t_embeddings.to(device))
            print("処理前")        
            cos_emb(s_embeddings.to("cpu"), t_embeddings.to("cpu"))

        # last hidden state
        src_embeddings.extend(s_out.to("cpu"))
        trg_embeddings.extend(t_out.to("cpu"))

    src_embeddings = [src_embeddings[idx] for idx in np.argsort(s_length_sorted_idx)]
    src_embeddings = torch.stack(src_embeddings)
    trg_embeddings = [trg_embeddings[idx] for idx in np.argsort(t_length_sorted_idx)]
    trg_embeddings = torch.stack(trg_embeddings)

    return src_embeddings, trg_embeddings


def main():
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    labse_model = AutoModel.from_pretrained(args.model)

    # base_model_en.to(device)
    # base_model_ja.to(device)

    print(args.model_path)
    # load lightning model
    base_model = MLP_sev.load_from_checkpoint(args.model_path)

    src_sentences = ["Details of dose rate of Fugen Power Plant can be calculated by using DERS software.", "The characteristics of R5 version of this software, instruction manual, and design document were summarized."]
    trg_sentences = ["ＤＥＲＳソフトウエアを用いて「ふげん発電所」の線量率を詳細に計算できる。", "このソフトウエアのＲ５バージョンの特徴，利用マニュアルと設計文書をまとめた。"]
    
    src_emb, trg_emb = embedding_pare(tokenizer, labse_model ,base_model,  src_sentences, trg_sentences, args.batch_size, device)
    print("処理後")
    cos_emb(src_emb, trg_emb )

if __name__ == "__main__":
    main()

