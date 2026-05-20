import torch
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def negative_sampling(example):
    if example['labels'] == 0:
        return random.random() > 0.7
    return True 


def compute_metrics(gold, predicted):
    p = precision_score(gold, predicted, zero_division=0)
    r = recall_score(gold, predicted, zero_division=0)
    f = f1_score(gold, predicted, zero_division=0)
    return p, r, f

def record_best_scores(timestamp, precision, recall, f1, filename):
    with open(filename, 'a') as file:
        file.write(f"{timestamp}\t{precision*100:.2f}\t{recall*100:.2f}\t{f1*100:.2f}\n")