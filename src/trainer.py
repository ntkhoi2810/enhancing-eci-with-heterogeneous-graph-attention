import torch
import torch.nn as nn
from src.utils import compute_metrics
from torch.amp import autocast, GradScaler

class ModelTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()

    def train_epoch(self, dataloader_mask, dataloader_tag):
        self.model.train()
        mean_loss = torch.zeros(1).to(self.device)
        predicted_all, gold_all = [], []
        
        for iteration, (mask_data, tag_data) in enumerate(zip(dataloader_mask, dataloader_tag)):
            mask_data = {k: v.to(self.device) for k, v in mask_data.items() if k != 'labels'}
            labels = tag_data['labels'].to(self.device)
            tag_data = {k: v.to(self.device) for k, v in tag_data.items() if k != 'labels'}
            
            outputs = self.model(mask_data, tag_data).squeeze(1)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            # self.optimizer.step()
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
    
    def evaluate(self, dataloader_mask_test, dataloader_tag_test):
        self.model.eval()
        mean_loss_test = 0
        predicted_all_test = []
        gold_all_test = []
        with torch.no_grad():
            for iteration, (mask_data, tag_data) in enumerate(zip(dataloader_mask_test, dataloader_tag_test)):
                labels = tag_data['labels'].to(self.device)
                mask_data = {k: v.to(self.device) for k, v in mask_data.items() if k != 'labels'}
                tag_data = {k: v.to(self.device) for k, v in tag_data.items() if k != 'labels'}
                
                outputs = self.model(mask_data, tag_data).squeeze(1)
                loss = self.criterion(outputs, labels)
                mean_loss_test = (mean_loss_test * iteration + loss.detach()) / (iteration + 1)

                predicted = torch.argmax(outputs, dim=-1)
                predicted_all_test += list(predicted.cpu().numpy())
                gold_all_test += list(labels.cpu().numpy())
        
        p, r, f1 = compute_metrics(gold_all_test, predicted_all_test)
        return p, r, f1, mean_loss_test.item()
