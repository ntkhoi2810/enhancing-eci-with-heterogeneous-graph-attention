import pandas as pd
import random
import re
from datasets import Dataset

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