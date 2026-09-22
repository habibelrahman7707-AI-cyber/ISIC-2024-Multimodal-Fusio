import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from .preprocess import safe_image_path

class ISICDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, meta_features=None):
        """
        df: dataframe with columns image_id (tile_name) and label and optional meta columns
        meta_features: list of metadata columns to use (e.g. ['age','sex','site'])
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.meta_features = meta_features or []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        
        img_path = safe_image_path(img_id)
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # metadata as vector (normalized in df preprocessing)
        if self.meta_features:
            meta = row[self.meta_features].astype(np.float32).values
            meta = torch.tensor(meta, dtype=torch.float)
        else:
            meta = torch.tensor([], dtype=torch.float)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return img, meta, label
