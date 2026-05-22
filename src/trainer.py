# import torch
# import torch.nn as nn
# from src.utils import compute_metrics
# from src.loss import FocalLoss
# from torch.amp import autocast, GradScaler
# from tqdm import tqdm

# class ModelTrainer:
#     def __init__(self, model, optimizer, device, class_weights=None, gamma=2.0):
#         self.model = model
#         self.optimizer = optimizer
#         self.device = device

#         if class_weights is not None:
#             class_weights = class_weights.to(device)
        
#         # self.criterion = nn.CrossEntropyLoss()
#         self.criterion = FocalLoss(weight=class_weights, gamma=gamma)
        
#         self.scaler = GradScaler()

#     def train_epoch(self, dataloader_mask, dataloader_tag, epoch_info):
#         self.model.train()
#         mean_loss = torch.zeros(1).to(self.device)
#         predicted_all, gold_all = [], []

#         pbar = tqdm(zip(dataloader_mask, dataloader_tag), total=len(dataloader_mask), desc=f"Train {epoch_info}", dynamic_ncols=True, leave=False)
        
#         for iteration, (mask_data, tag_data) in enumerate(pbar):
#             mask_data = {k: v.to(self.device) for k, v in mask_data.items() if k != 'labels'}
#             labels = tag_data['labels'].to(self.device)
#             tag_data = {k: v.to(self.device) for k, v in tag_data.items() if k != 'labels'}
            
#             with autocast('cuda'):
#                 outputs = self.model(mask_data, tag_data).squeeze(1)
#                 loss = self.criterion(outputs, labels)
            
#             self.optimizer.zero_grad()
            
#             # loss.backward()
#             # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
#             # self.optimizer.step()
#             self.scaler.scale(loss).backward()
#             self.scaler.unscale_(self.optimizer)
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
            
#             mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)
#             predicted_all += list(torch.argmax(outputs, dim=-1).cpu().numpy())
#             gold_all += list(labels.cpu().numpy())
            
#         p, r, f1 = compute_metrics(gold_all, predicted_all)
#         return p, r, f1, mean_loss.item()
    
#     def evaluate(self, dataloader_mask_test, dataloader_tag_test, epoch_info):
#         self.model.eval()
#         mean_loss_test = 0
#         predicted_all_test = []
#         gold_all_test = []

#         pbar = tqdm(zip(dataloader_mask_test, dataloader_tag_test), total=len(dataloader_mask_test), desc=f"Eval  {epoch_info}", dynamic_ncols=True, leave=False)
        
#         with torch.no_grad():
#             for iteration, (mask_data, tag_data) in enumerate(pbar):
#                 labels = tag_data['labels'].to(self.device)
#                 mask_data = {k: v.to(self.device) for k, v in mask_data.items() if k != 'labels'}
#                 tag_data = {k: v.to(self.device) for k, v in tag_data.items() if k != 'labels'}
                
#                 outputs = self.model(mask_data, tag_data).squeeze(1)
#                 loss = self.criterion(outputs, labels)
#                 mean_loss_test = (mean_loss_test * iteration + loss.detach()) / (iteration + 1)

#                 predicted = torch.argmax(outputs, dim=-1)
#                 predicted_all_test += list(predicted.cpu().numpy())
#                 gold_all_test += list(labels.cpu().numpy())
        
#         p, r, f1 = compute_metrics(gold_all_test, predicted_all_test)
#         return p, r, f1, mean_loss_test.item()


import torch
import torch.nn as nn
from src.utils import compute_metrics
from src.loss import FocalLoss
from torch.amp import autocast, GradScaler
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, optimizer, device, class_weights=None, gamma=2.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        if class_weights is not None:
            class_weights = class_weights.to(device)
        
        self.criterion = FocalLoss(weight=class_weights, gamma=gamma)
        self.scaler = GradScaler()

    def train_epoch(self, dataloader_mask, dataloader_tag, epoch_info):
        self.model.train()
        mean_loss = torch.zeros(1).to(self.device)
        predicted_all, gold_all = [], []

        pbar = tqdm(zip(dataloader_mask, dataloader_tag), total=len(dataloader_mask), desc=f"Train {epoch_info}", dynamic_ncols=True, leave=False)
        
        for iteration, (mask_data, tag_data) in enumerate(pbar):
            # ---> MỚI: Tách cấu trúc đồ thị ngữ pháp (Graph data) để xử lý riêng biệt
            graph_data = mask_data.pop('graph_data', None)
            if 'graph_data' in tag_data:
                tag_data.pop('graph_data') # Xóa bản sao thừa từ tag_data

            # Đẩy phần dữ liệu Text/Token sang GPU thiết bị huấn luyện
            mask_data = {k: v.to(self.device) for k, v in mask_data.items() if k != 'labels'}
            labels = tag_data['labels'].to(self.device)
            tag_data = {k: v.to(self.device) for k, v in tag_data.items() if k != 'labels'}
            
            with autocast('cuda'):
                # ---> MỚI: Truyền graph_data vào bước forward của Causal_Model
                outputs = self.model(mask_data, tag_data, graph_data).squeeze(1)
                loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)
            predicted_all += list(torch.argmax(outputs, dim=-1).cpu().numpy())
            gold_all += list(labels.cpu().numpy())
            
        p, r, f1 = compute_metrics(gold_all, predicted_all)
        return p, r, f1, mean_loss.item()
    
    def evaluate(self, dataloader_mask_test, dataloader_tag_test, epoch_info):
        self.model.eval()
        mean_loss_test = 0
        predicted_all_test = []
        gold_all_test = []

        pbar = tqdm(zip(dataloader_mask_test, dataloader_tag_test), total=len(dataloader_mask_test), desc=f"Eval  {epoch_info}", dynamic_ncols=True, leave=False)
        
        with torch.no_grad():
            for iteration, (mask_data, tag_data) in enumerate(pbar):
                # ---> MỚI: Tách cấu trúc đồ thị tương tự như khi huấn luyện
                graph_data = mask_data.pop('graph_data', None)
                if 'graph_data' in tag_data:
                    tag_data.pop('graph_data')

                labels = tag_data['labels'].to(self.device)
                mask_data = {k: v.to(self.device) for k, v in mask_data.items() if k != 'labels'}
                tag_data = {k: v.to(self.device) for k, v in tag_data.items() if k != 'labels'}
                
                # ---> MỚI: Truyền bổ sung thông tin đồ thị ngữ pháp phục vụ bước đánh giá Eval
                outputs = self.model(mask_data, tag_data, graph_data).squeeze(1)
                loss = self.criterion(outputs, labels)
                mean_loss_test = (mean_loss_test * iteration + loss.detach()) / (iteration + 1)

                predicted = torch.argmax(outputs, dim=-1)
                predicted_all_test += list(predicted.cpu().numpy())
                gold_all_test += list(labels.cpu().numpy())
        
        p, r, f1 = compute_metrics(gold_all_test, predicted_all_test)
        return p, r, f1, mean_loss_test.item()