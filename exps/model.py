import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from extract_config import extract_config


config = extract_config()

class ColaModel(nn.Module):
    def __init__(self, model_name = config["model_name"], lr = config["learning_rate"], eps = config["eps"]):
        super(ColaModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, int(config["model_layer"]["linear_output_dim"]))
        self.sigmoid = nn.Sigmoid()
        self.num_classes = int(config["model_layer"]["linear_output_dim"])
        self.lr = lr
        self.eps = eps
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = outputs.last_hidden_state[:, 0]
        x = self.W(h_cls)
        x = self.sigmoid(x)
        return x
    
    def training_step(self, batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = self.forward(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits = logits.to(torch.float32).reshape(-1)
        batch["label"] = batch["label"].to(torch.float32)
        train_loss = F.binary_cross_entropy(logits, batch["label"].to(device))
        preds = []
        for ele in logits:
            if ele > config["threshold"]:
                preds.append(1)
            else:
                preds.append(0)
        preds = np.array(preds)
        train_acc, train_pre_score, train_rec_score, train_f1 = self.calculate_metric(preds, batch["label"].cpu())
        return train_loss, train_acc, train_pre_score, train_rec_score, train_f1
    
    def validation_step(self, batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = self.forward(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits = logits.to(torch.float32).reshape(-1)
        batch["label"] = batch["label"].to(torch.float32)
        val_loss = F.binary_cross_entropy(logits, batch["label"].to(device))
        preds = []
        for ele in logits:
            if ele > config["threshold"]:
                preds.append(1)
            else:
                preds.append(0)
        preds = np.array(preds)
        val_acc, val_pre_score, val_rec_score, val_f1 = self.calculate_metric(preds, batch["label"].cpu())
        return val_loss, val_acc, val_pre_score, val_rec_score, val_f1, preds.tolist(), batch["label"].cpu().tolist()
    
    def calculate_metric(self, preds, targets):
        val_acc = torch.tensor(accuracy_score(preds, targets))
        pre_score = torch.tensor(precision_score(preds, targets))
        rec_score = torch.tensor(recall_score(preds, targets))
        f1 = torch.tensor(f1_score(preds, targets))
        return val_acc, pre_score, rec_score, f1
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = self.lr,  eps = self.eps)