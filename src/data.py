# import pandas as pd
# import random
# import re
# from datasets import Dataset

# def negative_sampling(example):
#     if example['labels'] == 0:
#         return random.random() > 0.7
#     return True

# def preprocess_row(row):
#     sent = str(row['sentence'])
#     e1 = str(row['e1'])
#     e2 = str(row['e2'])
    
#     def ireplace(text, old, new):
#         pattern = re.compile(re.escape(old), re.IGNORECASE)
#         return pattern.sub(new, text, count=1)
    
#     tagged = ireplace(sent, e1, f"<e1>{e1}</e1>")
#     tagged = ireplace(tagged, e2, f"<e2>{e2}</e2>")
    
#     to_mask = e1 if random.choice([True, False]) else e2
#     masked = ireplace(sent, to_mask, "<mask>")
    
#     if "<mask>" not in masked:
#         other_event = e2 if to_mask == e1 else e1
#         masked = ireplace(sent, other_event, "<mask>")
        
#     if "<mask>" not in masked:
#         masked = sent + " <mask>"
    
#     label_id = 1 if row['label_str'] == 'causal' else 0
#     return pd.Series([tagged, masked, label_id])

# def load_and_preprocess_data(dataset_path):
#     data = pd.read_pickle(dataset_path)
#     df = pd.DataFrame(data)
#     df.columns = ['id', 'sentence', 'e1', 'e2', 'label_str']
    
#     df[['event_tagged_sentence', 'event_masked_sentence', 'labels']] = df.apply(preprocess_row, axis=1)
#     df = df[['sentence', 'event_tagged_sentence', 'event_masked_sentence', 'e1', 'e2', 'labels']]
    
#     return Dataset.from_pandas(df)


import pandas as pd
import random
import re
import spacy # Cần cài đặt: pip install spacy && python -m spacy download en_core_web_sm
from datasets import Dataset

# Load mô hình phân tích cú pháp của spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Vui lòng tải model spaCy: python -m spacy download en_core_web_sm")

def negative_sampling(example):
    if example['labels'] == 0:
        return random.random() > 0.7
    return True

def extract_syntax_graph(sent, e1, e2):
    """
    Phân tích câu bằng spaCy để tạo đồ thị dị thể (Heterogeneous Graph).
    """
    doc = nlp(sent)
    edges = {}
    e1_idx, e2_idx = 0, 0
    
    for token in doc:
        if e1.lower() in token.text.lower(): e1_idx = token.i
        if e2.lower() in token.text.lower(): e2_idx = token.i
        
        # Chỉ sử dụng string làm key để PyArrow có thể lưu trữ được
        rel = token.dep_ if token.dep_ in ['nsubj', 'prep', 'pobj', 'dobj', 'amod', 'ROOT'] else 'other'
        
        if rel not in edges:
            edges[rel] = [[], []]
            
        edges[rel][0].append(token.head.i) # Source node
        edges[rel][1].append(token.i)      # Target node
        
    return edges, e1_idx, e2_idx

def preprocess_row(row):
    sent = str(row['sentence'])
    e1 = str(row['e1'])
    e2 = str(row['e2'])
    
    # [Giữ nguyên logic tạo tagged và masked của bạn...]
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
    
    # ---> MỚI: Trích xuất đồ thị ngữ pháp
    syntax_edges, e1_idx, e2_idx = extract_syntax_graph(sent, e1, e2)
    
    return pd.Series([tagged, masked, label_id, syntax_edges, e1_idx, e2_idx])

def load_and_preprocess_data(dataset_path):
    data = pd.read_pickle(dataset_path)
    df = pd.DataFrame(data)
    df.columns = ['id', 'sentence', 'e1', 'e2', 'label_str']
    
    # Cập nhật khung chứa cho dataframe
    df[['event_tagged_sentence', 'event_masked_sentence', 'labels', 'syntax_edges', 'e1_idx', 'e2_idx']] = df.apply(preprocess_row, axis=1)
    
    # Giữ lại các cột cần thiết cho model
    df = df[['sentence', 'event_tagged_sentence', 'event_masked_sentence', 'e1', 'e2', 'labels', 'syntax_edges', 'e1_idx', 'e2_idx']]
    
    return Dataset.from_pandas(df)