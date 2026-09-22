import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE

def get_train_transforms():
    return A.Compose([
        A.SmallestMaxSize(max_size=IMG_SIZE+20),
        A.RandomCrop(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.12, rotate_limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])
