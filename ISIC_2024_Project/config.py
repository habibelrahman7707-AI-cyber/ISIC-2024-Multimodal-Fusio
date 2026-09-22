import os
import torch
import random
import numpy as np

# ---------------------------
# Config & Hyperparameters
# ---------------------------
DATA_DIR = "data/ISIC2024"
IMG_DIR = os.path.join(DATA_DIR, "images")
META_CSV = os.path.join(DATA_DIR, "metadata.csv")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

SEED = 42

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
