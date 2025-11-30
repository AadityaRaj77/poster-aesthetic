import os
import sys
from ava_dataset import AVADataset
from transforms import train_transforms
from torch.utils.data import DataLoader

CSV = r"../data/ava_subset/labels.csv"
IMG_DIR = r"../data/ava_subset/images"

if not os.path.exists(CSV):
    raise SystemExit(f"labels.csv not found at {CSV}")

dataset = AVADataset(csv_path=CSV, img_dir=IMG_DIR, transform=train_transforms)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

for imgs, labels in loader:
    print("Batch images shape:", imgs.shape)   # should be [4, 3, 224, 224]
    print("Batch labels:", labels)
    break
