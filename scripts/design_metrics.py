import cv2
import numpy as np
from PIL import Image

# Convert PIL to OpenCV format
def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 1. Contrast (luminance variance)
def compute_contrast(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return float(np.var(gray))

# 2. Clutter (edge density)
def compute_clutter(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.mean(edges > 0))

# 3. Color Harmony (Hue variance)
def compute_color_harmony(cv_img):
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    return float(np.var(h))

# 4. Balance (center of mass distance)
def compute_balance(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    M = cv2.moments(gray)
    if M["m00"] == 0:
        return 0.0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    h, w = gray.shape
    return float(np.sqrt((cx - w/2)**2 + (cy - h/2)**2))

# 5. Whitespace (low texture regions)
def compute_whitespace(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(lap)
    low_texture = texture < 5
    return float(np.mean(low_texture))

# Main function called by notebook
def compute_all_design_metrics(pil_img):
    cv_img = pil_to_cv(pil_img)

    metrics = {
        "contrast": compute_contrast(cv_img),
        "clutter": compute_clutter(cv_img),
        "color_harmony": compute_color_harmony(cv_img),
        "balance": compute_balance(cv_img),
        "whitespace": compute_whitespace(cv_img)
    }

    # Normalize each metric to [0,1]
    # To avoid zero-division and keep simple scaling
    norm_metrics = {k: v / (v + 1e-6) for k, v in metrics.items()}

    # Final score = average of metrics
    design_score = float(np.mean(list(norm_metrics.values())))

    return design_score, norm_metrics
