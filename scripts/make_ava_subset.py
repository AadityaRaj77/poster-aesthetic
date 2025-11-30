import os
import pandas as pd
from shutil import copyfile


FULL_LABELS = r"Z:\Jupyter\Pytorch\data\ava\AVA.txt"
FULL_IMAGES = r"Z:\Jupyter\Pytorch\data\ava\images"
OUT_DIR = r"Z:\Jupyter\Pytorch\data\ava_subset"


os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

# AVA.txt format:
# image_id, challenge_id, c1..c10 votes, extra columns...
df = pd.read_csv(FULL_LABELS, sep=" ", header=None)

# Keep only first 12 columns: image_id, challenge, votes(1-10)
df = df.iloc[:, :12]
df.columns = [
    "image_id", "challenge",
    "c1", "c2", "c3", "c4", "c5",
    "c6", "c7", "c8", "c9", "c10"
]

vote_cols = ["c1", "c2", "c3", "c4", "c5",
             "c6", "c7", "c8", "c9", "c10"]

# Remove rows with zero total votes
df["total_votes"] = df[vote_cols].sum(axis=1)
df = df[df["total_votes"] > 0]

# Compute weighted aesthetic score
df["score"] = sum(
    (i + 1) * df[vote_cols].iloc[:, i] for i in range(10)
) / df["total_votes"]

# Pick random subset of 1000 images (change number if needed)
subset = df.sample(1000, random_state=42)

rows = []

for _, row in subset.iterrows():
    img_name = f"{int(row['image_id'])}.jpg"
    src = os.path.join(FULL_IMAGES, img_name)
    dst = os.path.join(OUT_DIR, "images", img_name)

    if os.path.exists(src):
        copyfile(src, dst)
        rows.append({"image": img_name, "score": row["score"]})

labels_path = os.path.join(OUT_DIR, "labels.csv")
pd.DataFrame(rows).to_csv(labels_path, index=False)

print("AVA subset created successfully!")
print(f"Images saved in: {OUT_DIR}\\images")
print(f"Labels saved in: {labels_path}")
