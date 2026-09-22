import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from config import DEVICE

def train_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    for imgs, metas, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        metas = metas.to(DEVICE, non_blocking=True) if metas.nelement() != 0 else None

        optimizer.zero_grad()
        with autocast():
            logits = model(imgs, metas)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion):
    model.eval()
    preds = []
    trues = []
    losses = []
    with torch.no_grad():
        for imgs, metas, labels in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            metas = metas.to(DEVICE, non_blocking=True) if metas.nelement() != 0 else None

            with autocast():
                logits = model(imgs, metas)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            preds.extend(probs.tolist())
            trues.extend(labels.detach().cpu().numpy().tolist())
            losses.append(loss.item())
            
    auc = roc_auc_score(trues, preds) if len(set(trues)) > 1 else 0.5
    return np.mean(losses), auc, np.array(preds), np.array(trues)
