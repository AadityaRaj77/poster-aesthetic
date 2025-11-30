import torch
import torch.nn as nn
import pandas as pd
import cv2
import os
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = "../data/ava_subset/images"
DESIGN_CSV = "../results/design_scores.csv"
OUT_CSV = "../results/final_scores.csv"
MODEL_PATH = "../results/resnet18_final.pt"

# Load Model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Design Data
df = pd.read_csv(DESIGN_CSV)

final_rows = []

for idx, row in df.iterrows():
    fname = row["image"]
    img_path = os.path.join(IMG_DIR, fname)

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cnn_score = model(tensor).item()

    design_score = row["design_score"]

    # Score Fusion
    final_score = 0.6 * cnn_score + 0.4 * (design_score * 10)

    final_rows.append([
        fname,
        cnn_score,
        design_score,
        final_score
    ])

final_df = pd.DataFrame(final_rows, columns=[
    "image",
    "cnn_score",
    "design_score",
    "final_aesthetic_score"
])

final_df.to_csv(OUT_CSV, index=False)
print("Saved fused scores to:", OUT_CSV)
