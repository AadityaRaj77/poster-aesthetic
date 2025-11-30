import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
from ava_dataset import AVADataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Load Model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("../results/resnet18_final.pt", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

#Hook for Grad-CAM
features = []
grads = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    grads.append(grad_out[0])

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

#Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#Load One Test Image
IMG_PATH = r"../data/ava_subset/images"  
IMG_NAME = "poster2.png"         
img = Image.open(f"{IMG_PATH}/{IMG_NAME}").convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

#Forward Pass
pred = model(input_tensor)
model.zero_grad()

#Backward for Grad-CAM
pred.backward()

#Compute Grad-CAM 
fmap = features[0]      # [1, C, H, W]
grad = grads[0]         # [1, C, H, W]

weights = grad.mean(dim=(2,3), keepdim=True)
cam = (weights * fmap).sum(dim=1).squeeze()
cam = F.relu(cam)

cam = cam.detach().cpu().numpy()
cam = cv2.resize(cam, (224, 224))
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

#Overlay Heatmap
orig = cv2.imread(f"{IMG_PATH}/{IMG_NAME}")
orig = cv2.resize(orig, (224, 224))

heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

cv2.imwrite("../results/gradcam_overlay.png", overlay)
print("Saved Grad-CAM to results/gradcam_overlay.png")
