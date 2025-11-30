import cv2
import numpy as np
import os
import pandas as pd

IMG_DIR = "../data/ava_subset/images"
OUT_CSV = "../results/design_scores.csv"

def color_harmony_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    std = np.std(h)
    return 1.0 - min(std / 90.0, 1.0)

def contrast_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    return min(std / 70.0, 1.0)

def clutter_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.sum(edges > 0) / edges.size
    return 1.0 - min(edge_density * 10, 1.0)

def balance_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    left = gray[:, :w//2]
    right = gray[:, w//2:]
    diff = abs(left.mean() - right.mean())
    return 1.0 - min(diff / 50.0, 1.0)

def whitespace_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = gray < 220
    empty_ratio = 1.0 - (np.sum(thresh) / thresh.size)
    return min(empty_ratio * 1.5, 1.0)

def compute_design_score(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    c1 = color_harmony_score(img)
    c2 = contrast_score(img)
    c3 = clutter_score(img)
    c4 = balance_score(img)
    c5 = whitespace_score(img)

    final_score = 0.25*c1 + 0.25*c2 + 0.2*c3 + 0.15*c4 + 0.15*c5

    return c1, c2, c3, c4, c5, final_score

def main():
    rows = []

    for fname in os.listdir(IMG_DIR):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(IMG_DIR, fname)
            c1, c2, c3, c4, c5, f = compute_design_score(path)

            rows.append([fname, c1, c2, c3, c4, c5, f])

    df = pd.DataFrame(rows, columns=[
        "image",
        "color_harmony",
        "contrast",
        "clutter",
        "balance",
        "whitespace",
        "design_score"
    ])

    df.to_csv(OUT_CSV, index=False)
    print("Saved design metrics to:", OUT_CSV)

if __name__ == "__main__":
    main()
