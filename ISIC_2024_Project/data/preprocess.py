import pandas as pd
from pathlib import Path
from config import IMG_DIR

def safe_image_path(img_id):
    """
    Safely locate the image file.
    """
    p1 = Path(IMG_DIR) / f"{img_id}.jpg"
    p2 = Path(IMG_DIR) / img_id
    if p1.exists():
        return str(p1)
    if p2.exists():
        return str(p2)
    
    candidate = list(Path(IMG_DIR).rglob(f"{img_id}*.jpg"))
    if candidate:
        return str(candidate[0])
    raise FileNotFoundError(f"Image {img_id} not found in {IMG_DIR}")

def prepare_dataframe(meta_csv_path):
    """
    Load CSV, standardize columns, encode metadata, and return cleaned dataframe.
    """
    df = pd.read_csv(meta_csv_path)
    
    # Standardize column names
    if "tile_name" in df.columns:
        df = df.rename(columns={"tile_name": "image_id"})
    elif "image" in df.columns:
        df = df.rename(columns={"image": "image_id"})
        
    if "malignancy" in df.columns:
        df = df.rename(columns={"malignancy": "label"})
    elif "label" in df.columns:
        pass
    elif "target" in df.columns:
        df = df.rename(columns={"target": "label"})
    else:
        raise ValueError("Label column not found; please make sure metadata.csv has label column (malignancy/label/target)")

    # Keep only images that exist
    def exists_row(x):
        try:
            safe_image_path(x)
            return True
        except FileNotFoundError:
            return False

    df["exists"] = df["image_id"].apply(exists_row)
    df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)

    # Process metadata features
    meta_features = []
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
        df["age_norm"] = (df["age"] - df["age"].mean()) / (df["age"].std() + 1e-9)
        meta_features.append("age_norm")

    if "sex" in df.columns:
        df["sex_enc"] = df["sex"].map({"male": 0, "female": 1}).fillna(-1)
        df["sex_enc"] = df["sex_enc"].astype(float)
        df["sex_enc"] = (df["sex_enc"] - df["sex_enc"].mean()) / (df["sex_enc"].std() + 1e-9)
        meta_features.append("sex_enc")

    if "site" in df.columns:
        df["site_le"] = pd.factorize(df["site"])[0].astype(float)
        df["site_le"] = (df["site_le"] - df["site_le"].mean()) / (df["site_le"].std() + 1e-9)
        meta_features.append("site_le")

    # keep only required columns
    keep_cols = ["image_id", "label"] + meta_features
    df = df[keep_cols]
    df["label"] = df["label"].astype(int)
    
    return df, meta_features
