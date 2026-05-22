# import argparse
# import os
# import datetime
# import torch
# import numpy as np 
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, DataCollatorWithPadding
# from tqdm import tqdm

# from src.utils import setup_seed, record_best_scores, EarlyStopping
# from src.data import load_and_preprocess_data, negative_sampling
# from src.models import Causal_Model
# from src.trainer import ModelTrainer


# def main(args):
#     setup_seed(args.SEED)
#     torch.backends.cudnn.benchmark = True
    
#     tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
#     special_tokens = ['<e1>', '</e1>', '<e2>', '</e2>']
#     tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
#     total_dataset = load_and_preprocess_data(args.dataset)
    
#     # if args.dataset_name == 'ESC_star' and args.shuffle:
#     total_dataset = total_dataset.shuffle(seed=args.SEED)
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f'\nDevice: {device}')

#     fold_size = len(total_dataset) // args.num_folds
#     checkpoint_path = f'checkpoints/{args.dataset_name}_{args.bert_path.split("/")[-1]}'
#     os.makedirs(checkpoint_path, exist_ok=True)
#     print(f"Saved checkpoint at: {checkpoint_path}")

#     for i in range(args.num_folds):
#         print(f"\n=== Training | Fold {i+1}/{args.num_folds} ===")

#         test_indices = list(range(i * fold_size, (i + 1) * fold_size))
#         train_indices = list(set(range(len(total_dataset))) - set(test_indices))
        
#         train_fold = total_dataset.select(train_indices).shuffle(seed=args.SEED)

#         # if args.dataset_name == 'CTB':
#         #     # train_fold = train_fold.shuffle(seed=args.SEED).filter(negative_sampling)
#         #     train_fold = train_fold.filter(negative_sampling)
            
#         test_fold = total_dataset.select(test_indices)

#         labels = train_fold['labels']
#         count_0 = labels.count(0)
#         count_1 = labels.count(1)
#         total_samples = len(labels)

#         weight_0 = total_samples / (2.0 * count_0)
#         weight_1 = total_samples / (2.0 * count_1)
#         class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)

#         print(f"[*] Class Weights Fold {i+1}: Normal (0) = {weight_0:.4f}, Causal (1) = {weight_1:.4f}")

#         def tokenize_col(text_column):
#             return lambda x: tokenizer(x[text_column], truncation=True)

#         cols_to_remove = ['sentence', 'event_tagged_sentence', 'event_masked_sentence', 'e1', 'e2']
        
#         masked_train = train_fold.map(
#             tokenize_col("event_masked_sentence"), 
#             batched=True, batch_size=32
#         ).remove_columns(cols_to_remove)
        
#         tagged_train = train_fold.map(
#             tokenize_col("event_tagged_sentence"), 
#             batched=True, batch_size=32
#         ).remove_columns(cols_to_remove)
        
#         masked_train.set_format("torch")
#         tagged_train.set_format("torch")

#         masked_test = test_fold.map(
#             tokenize_col("event_masked_sentence"), 
#             batched=True, batch_size=32
#             ).remove_columns(cols_to_remove)
        
#         tagged_test = test_fold.map(
#             tokenize_col("event_tagged_sentence"), 
#             batched=True, batch_size=32
#         ).remove_columns(cols_to_remove)
        
#         masked_test.set_format("torch")
#         tagged_test.set_format("torch")

#         data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#         # dataloader_mask_train = DataLoader(masked_train, shuffle=False, batch_size=args.train_batchsize, collate_fn=data_collator)
#         # dataloader_tag_train = DataLoader(tagged_train, shuffle=False, batch_size=args.train_batchsize, collate_fn=data_collator)
#         # dataloader_mask_test = DataLoader(masked_test, shuffle=False, batch_size=args.test_batchsize, collate_fn=data_collator)
#         # dataloader_tag_test = DataLoader(tagged_test, shuffle=False, batch_size=args.test_batchsize, collate_fn=data_collator)
        
#         dataloader_mask_train = DataLoader(
#             masked_train, 
#             shuffle=False, 
#             batch_size=args.train_batchsize, 
#             collate_fn=data_collator, 
#             num_workers=4, 
#             pin_memory=True
#         )
#         dataloader_tag_train = DataLoader(
#             tagged_train, 
#             shuffle=False, 
#             batch_size=args.train_batchsize, 
#             collate_fn=data_collator, 
#             num_workers=4, 
#             pin_memory=True
#         )
#         dataloader_mask_test = DataLoader(
#             masked_test, 
#             shuffle=False, 
#             batch_size=args.test_batchsize, 
#             collate_fn=data_collator, 
#             num_workers=4, 
#             pin_memory=True
#         )
#         dataloader_tag_test = DataLoader(
#             tagged_test, 
#             shuffle=False, 
#             batch_size=args.test_batchsize, 
#             collate_fn=data_collator, 
#             num_workers=4, 
#             pin_memory=True
#         )

#         # dataloader_mask_train = tqdm(dataloader_mask_train, dynamic_ncols=True)
#         # dataloader_mask_test = tqdm(dataloader_mask_test, dynamic_ncols=True)

#         model = Causal_Model(
#             bert_path=args.bert_path, 
#             d_model=args.d_model, 
#             num_heads=args.num_heads, 
#             dropout_rate=args.dropout_rate, 
#             device=device, 
#             special_tokens=special_tokens, 
#             visualize=args.visualize
#         ).to(device)
        
#         optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
#         trainer = ModelTrainer(model, optimizer, device, class_weights, gamma=args.focal_gamma)


#         early_stopping = EarlyStopping(patience=args.patience, verbose=True)

#         for epoch in range(args.num_epochs):

#             ep_info = f"[Ep {epoch+1}/{args.num_epochs}]"
            
#             train_p, train_r, train_f1, train_loss = trainer.train_epoch(dataloader_mask_train, dataloader_tag_train, ep_info)
#             test_p, test_r, test_f1, test_loss = trainer.evaluate(dataloader_mask_test, dataloader_tag_test, ep_info)
            
#             print(f"[Epoch {epoch+1}], loss: {train_loss}")
#             print(f"Training validation:")
#             print(f"p: {test_p:.4f}, r: {test_r:.4f}, f1: {test_f1:.4f}")
            
#             is_new_best = early_stopping(test_f1 * 100)
        
#             if is_new_best:
#                 torch.save(model.state_dict(), os.path.join(checkpoint_path, f'best_model_fold{i+1}.pt'))
#                 print(f"-> NEW BEST F1: {test_f1*100:.2f}%. CHECKPOINT SAVED!")
#                 current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                 record_best_scores(
#                     current_time, 
#                     test_p, 
#                     test_r, 
#                     test_f1, 
#                     os.path.join(checkpoint_path, f'best_scores_fold{i+1}.txt')
#                 )
                
#             if early_stopping.early_stop:
#                 print(f"\n[!] EARLY STOPPING TRIGGED | FOLD {i+1}!")
#                 break
        
#         print(f"=== END OF TRAINING | FOLD {i+1} ===\n")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='data/ESC_dataset')
#     parser.add_argument('--dataset_name', type=str, default='ESC')
#     parser.add_argument('--num_folds', type=int, default=5)
#     parser.add_argument('--num_epochs', type=int, default=50)
#     parser.add_argument('--train_batchsize', type=int, default=20)
#     parser.add_argument('--patience', type=int, default=7)
#     parser.add_argument('--test_batchsize', type=int, default=20)
#     parser.add_argument('--learning_rate', type=float, default=1e-5)
#     parser.add_argument('--bert_path', type=str, default='FacebookAI/roberta-large')
#     parser.add_argument('--d_model', type=int, default=1024)
#     parser.add_argument('--num_heads', type=int, default=16)
#     parser.add_argument('--dropout_rate', type=float, default=0.5)
#     parser.add_argument('--visualize', action='store_true')
#     parser.add_argument('--SEED', type=int, default=3407)
#     parser.add_argument('--shuffle', action='store_true')
#     parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma value for Focal Loss')

#     args = parser.parse_args()
#     main(args)



import argparse
import os
import datetime
import torch
import numpy as np 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm

from src.utils import setup_seed, record_best_scores, EarlyStopping
from src.data import load_and_preprocess_data, negative_sampling
from src.models import Causal_Model
from src.trainer import ModelTrainer

# ---> MỚI: Định nghĩa Bộ Collate tùy chỉnh nhằm gom cụm dữ liệu đồ thị ngữ pháp và văn bản vào cùng 1 Batch
class GraphTextCollate:
    def __init__(self, tokenizer):
        self.text_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    def __call__(self, features):
        # Tách cấu trúc đồ thị (edges, e1_idx, e2_idx) ra khỏi tập tính năng thông thường
        graph_features = {
            'edges': [f['syntax_edges'] for f in features],
            'e1_idx': [f['e1_idx'] for f in features],
            'e2_idx': [f['e2_idx'] for f in features]
        }
        
        # Chỉ giữ lại các token IDs và labels dành cho mô hình ngôn ngữ
        text_features = []
        for f in features:
            text_f = {k: v for k, v in f.items() if k not in ['syntax_edges', 'e1_idx', 'e2_idx']}
            text_features.append(text_f)
            
        # Pading văn bản tự động sang dạng PyTorch Tensor
        batch = self.text_collator(text_features)
        
        # Đính kèm dữ liệu đồ thị vào Batch kết quả
        batch['graph_data'] = graph_features
        return batch


def main(args):
    setup_seed(args.SEED)
    torch.backends.cudnn.benchmark = True
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    special_tokens = ['<e1>', '</e1>', '<e2>', '</e2>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    total_dataset = load_and_preprocess_data(args.dataset)
    total_dataset = total_dataset.shuffle(seed=args.SEED)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nDevice: {device}')

    fold_size = len(total_dataset) // args.num_folds
    checkpoint_path = f'checkpoints/{args.dataset_name}_{args.bert_path.split("/")[-1]}'
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Saved checkpoint at: {checkpoint_path}")

    for i in range(args.num_folds):
        print(f"\n=== Training | Fold {i+1}/{args.num_folds} ===")

        test_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = list(set(range(len(total_dataset))) - set(test_indices))
        
        train_fold = total_dataset.select(train_indices).shuffle(seed=args.SEED)
        test_fold = total_dataset.select(test_indices)

        labels = train_fold['labels']
        count_0 = labels.count(0)
        count_1 = labels.count(1)
        total_samples = len(labels)

        weight_0 = total_samples / (2.0 * count_0)
        weight_1 = total_samples / (2.0 * count_1)
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)

        print(f"[*] Class Weights Fold {i+1}: Normal (0) = {weight_0:.4f}, Causal (1) = {weight_1:.4f}")

        def tokenize_col(text_column):
            return lambda x: tokenizer(x[text_column], truncation=True)

        cols_to_remove = ['sentence', 'event_tagged_sentence', 'event_masked_sentence', 'e1', 'e2']
        
        masked_train = train_fold.map(
            tokenize_col("event_masked_sentence"), 
            batched=True, batch_size=32
        ).remove_columns(cols_to_remove)
        
        tagged_train = train_fold.map(
            tokenize_col("event_tagged_sentence"), 
            batched=True, batch_size=32
        ).remove_columns(cols_to_remove)
        
        # ---> LƯU Ý MỚI: Bỏ việc gọi .set_format("torch") trực tiếp tại đây, vì hàm collate_fn 
        # tự xử lý chuyển đổi kiểu dữ liệu cho từng loại trường thích hợp, tránh mất cấu trúc Graph dict.
        # masked_train.set_format("torch")
        # tagged_train.set_format("torch")

        masked_test = test_fold.map(
            tokenize_col("event_masked_sentence"), 
            batched=True, batch_size=32
        ).remove_columns(cols_to_remove)
        
        tagged_test = test_fold.map(
            tokenize_col("event_tagged_sentence"), 
            batched=True, batch_size=32
        ).remove_columns(cols_to_remove)
        
        # masked_test.set_format("torch")
        # tagged_test.set_format("torch")

        # Sử dụng Bộ dữ liệu tùy chỉnh mới thay thế cho DataCollatorWithPadding gốc
        data_collator = GraphTextCollate(tokenizer=tokenizer)
        
        dataloader_mask_train = DataLoader(
            masked_train, 
            shuffle=False, 
            batch_size=args.train_batchsize, 
            collate_fn=data_collator, 
            num_workers=4, 
            pin_memory=True
        )
        dataloader_tag_train = DataLoader(
            tagged_train, 
            shuffle=False, 
            batch_size=args.train_batchsize, 
            collate_fn=data_collator, 
            num_workers=4, 
            pin_memory=True
        )
        dataloader_mask_test = DataLoader(
            masked_test, 
            shuffle=False, 
            batch_size=args.test_batchsize, 
            collate_fn=data_collator, 
            num_workers=4, 
            pin_memory=True
        )
        dataloader_tag_test = DataLoader(
            tagged_test, 
            shuffle=False, 
            batch_size=args.test_batchsize, 
            collate_fn=data_collator, 
            num_workers=4, 
            pin_memory=True
        )

        model = Causal_Model(
            bert_path=args.bert_path, 
            d_model=args.d_model, 
            num_heads=args.num_heads, 
            dropout_rate=args.dropout_rate, 
            device=device, 
            special_tokens=special_tokens, 
            visualize=args.visualize
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        trainer = ModelTrainer(model, optimizer, device, class_weights, gamma=args.focal_gamma)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        for epoch in range(args.num_epochs):
            ep_info = f"[Ep {epoch+1}/{args.num_epochs}]"
            
            train_p, train_r, train_f1, train_loss = trainer.train_epoch(dataloader_mask_train, dataloader_tag_train, ep_info)
            test_p, test_r, test_f1, test_loss = trainer.evaluate(dataloader_mask_test, dataloader_tag_test, ep_info)
            
            print(f"[Epoch {epoch+1}], loss: {train_loss}")
            print(f"Training validation:")
            print(f"p: {test_p:.4f}, r: {test_r:.4f}, f1: {test_f1:.4f}")
            
            is_new_best = early_stopping(test_f1 * 100)
        
            if is_new_best:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, f'best_model_fold{i+1}.pt'))
                print(f"-> NEW BEST F1: {test_f1*100:.2f}%. CHECKPOINT SAVED!")
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                record_best_scores(
                    current_time, 
                    test_p, 
                    test_r, 
                    test_f1, 
                    os.path.join(checkpoint_path, f'best_scores_fold{i+1}.txt')
                )
                
            if early_stopping.early_stop:
                print(f"\n[!] EARLY STOPPING TRIGGED | FOLD {i+1}!")
                break
        
        print(f"=== END OF TRAINING | FOLD {i+1} ===\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/ESC_dataset')
    parser.add_argument('--dataset_name', type=str, default='ESC')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batchsize', type=int, default=20)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--test_batchsize', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--bert_path', type=str, default='FacebookAI/roberta-large')
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--SEED', type=int, default=3407)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma value for Focal Loss')

    args = parser.parse_args()
    main(args)