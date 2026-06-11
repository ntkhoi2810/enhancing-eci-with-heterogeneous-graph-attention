import pandas as pd
import random
import re
import numpy as np
import torch
import spacy
from datasets import Dataset
from transformers import DataCollatorWithPadding

# Tải mô hình NLP (tải 1 lần để tối ưu tốc độ)
nlp = spacy.load("en_core_web_sm")

def negative_sampling(example):
    if example['labels'] == 0:
        return random.random() > 0.7
    return True

def preprocess_row(row):
    sent = str(row['sentence'])
    e1 = str(row['e1'])
    e2 = str(row['e2'])
    
    def ireplace(text, old, new):
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        return pattern.sub(new, text, count=1)
    
    tagged = ireplace(sent, e1, f"<e1>{e1}</e1>")
    tagged = ireplace(tagged, e2, f"<e2>{e2}</e2>")
    
    to_mask = e1 if random.choice([True, False]) else e2
    masked = ireplace(sent, to_mask, "<mask>")
    
    if "<mask>" not in masked:
        other_event = e2 if to_mask == e1 else e1
        masked = ireplace(sent, other_event, "<mask>")
        
    if "<mask>" not in masked:
        masked = sent + " <mask>"
    
    label_id = 1 if row['label_str'] == 'causal' else 0
    return pd.Series([tagged, masked, label_id])

def load_and_preprocess_data(dataset_path):
    data = pd.read_pickle(dataset_path)
    df = pd.DataFrame(data)
    df.columns = ['id', 'sentence', 'e1', 'e2', 'label_str']
    
    df[['event_tagged_sentence', 'event_masked_sentence', 'labels']] = df.apply(preprocess_row, axis=1)
    df = df[['sentence', 'event_tagged_sentence', 'event_masked_sentence', 'e1', 'e2', 'labels']]
    
    return Dataset.from_pandas(df)


def tokenize_and_build_graph(batch, tokenizer, text_column, max_length=256):
    texts = batch[text_column]
    # Yêu cầu dùng Fast Tokenizer để có hàm word_ids()
    encodings = tokenizer(texts, truncation=True, max_length=max_length)
    
    dep_adjs = []
    seq_adjs = []
    
    for i, text in enumerate(texts):
        doc = nlp(text)
        seq_len = len(encodings.input_ids[i])
        word_ids = encodings.word_ids(batch_index=i)
        
        # Khởi tạo 2 ma trận kề
        dep_adj = np.zeros((seq_len, seq_len), dtype=np.float32)
        seq_adj = np.zeros((seq_len, seq_len), dtype=np.float32)
        
        # Ánh xạ word index (của spaCy) -> danh sách các subword index (của BERT)
        word_to_subwords = {}
        for sub_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:  # Bỏ qua các special tokens [CLS], [SEP]
                if word_idx not in word_to_subwords:
                    word_to_subwords[word_idx] = []
                word_to_subwords[word_idx].append(sub_idx)
        
        # 1. Xây dựng Dependency Graph
        for token in doc:
            u_word_idx = token.i
            v_word_idx = token.head.i
            
            if u_word_idx in word_to_subwords and v_word_idx in word_to_subwords:
                for u in word_to_subwords[u_word_idx]:
                    for v in word_to_subwords[v_word_idx]:
                        dep_adj[u, v] = 1.0
                        dep_adj[v, u] = 1.0  # Graph vô hướng
                        
        np.fill_diagonal(dep_adj, 1.0)  # Self-loop cho Dependency
        
        # 2. Xây dựng Sequential Graph (kết nối token kề nhau)
        for j in range(seq_len):
            seq_adj[j, j] = 1.0
            if j > 0:
                seq_adj[j, j-1] = 1.0
                seq_adj[j-1, j] = 1.0
                
        # Phải lưu dạng list of lists để Dataset tương thích lưu trữ
        dep_adjs.append(dep_adj.tolist())
        seq_adjs.append(seq_adj.tolist())
        
    encodings['dep_adj'] = dep_adjs
    encodings['seq_adj'] = seq_adjs
    return encodings


class GraphDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Tách graph_data ra trước khi HF xử lý (vì HF sẽ văng lỗi nếu pad 2D array)
        has_graph = 'dep_adj' in features[0]
        
        if has_graph:
            dep_adjs = [f.pop('dep_adj') for f in features]
            seq_adjs = [f.pop('seq_adj') for f in features]
            
        # Gọi collator mặc định để pad input_ids, attention_mask...
        batch = super().__call__(features)
        
        if has_graph:
            batch_size = len(features)
            max_len = batch['input_ids'].shape[1]  # Độ dài tối đa trong batch này
            
            padded_dep = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
            padded_seq = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
            
            for i in range(batch_size):
                seq_len = min(len(dep_adjs[i]), max_len)
                dep_tensor = torch.tensor(dep_adjs[i])[:seq_len, :seq_len]
                seq_tensor = torch.tensor(seq_adjs[i])[:seq_len, :seq_len]
                
                padded_dep[i, :seq_len, :seq_len] = dep_tensor
                padded_seq[i, :seq_len, :seq_len] = seq_tensor
            
            # Gói lại cấu trúc y như model kỳ vọng (`graph_data['adjs']`)
            batch['graph_data'] = {
                'adjs': [padded_dep, padded_seq]
            }
            
        return batch