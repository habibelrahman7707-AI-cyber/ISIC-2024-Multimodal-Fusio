import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc

from config import *
from data.preprocess import prepare_dataframe
from data.dataset import ISICDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.fusion import FusionModel
from train import train_epoch, validate_epoch

def main():
    print("Loading metadata...")
    df, meta_features = prepare_dataframe(META_CSV)
    print(f"Total samples found: {len(df)}. Using meta features: {meta_features}")

    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)

    train_ds = ISICDataset(train_df, IMG_DIR, transforms=get_train_transforms(), meta_features=meta_features)
    val_ds = ISICDataset(val_df, IMG_DIR, transforms=get_val_transforms(), meta_features=meta_features)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    meta_dim = len(meta_features)
    model = FusionModel(backbone_name="tf_efficientnetv2_s", meta_in_dim=meta_dim, num_classes=2, pretrained=True).to(DEVICE)

    counts = train_df["label"].value_counts().to_dict()
    print("Class distribution (train):", counts)
    total = len(train_df)
    pos_weight = (total - counts.get(1, 0)) / max(1, counts.get(1, 0))
    print("Pos weight (approx):", pos_weight)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n---- Epoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion)
        val_loss, val_auc, preds, trues = validate_epoch(model, val_loader, criterion)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        scheduler.step()
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_isic2024_fusion.pth"))
            print("Saved best model with AUC:", best_auc)

    print("\n=== Final evaluation on validation set ===")
    preds_bin = (preds > 0.5).astype(int)
    print(classification_report(trues, preds_bin, digits=4))
    fpr, tpr, _ = roc_curve(trues, preds)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC:", roc_auc)
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend(); plt.title("Validation ROC")
    plt.savefig(os.path.join(OUT_DIR, "roc_val.png"))
    print(f"Saved ROC to {OUT_DIR}/roc_val.png")

if __name__ == "__main__":
    main()
