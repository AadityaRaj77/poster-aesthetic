import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt

from ava_dataset import AVADataset
from transforms import train_transforms, val_transforms

def train(csv_path, img_dir, out_dir, epochs=5, batch_size=16, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    dataset = AVADataset(csv_path=csv_path, img_dir=img_dir, transform=train_transforms)
    # simple split: 90% train 10% val
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)

        epoch_train_loss = running / (len(train_loader.dataset))
        train_losses.append(epoch_train_loss)

        # validation
        model.eval()
        vrunning = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1).float()
                preds = model(imgs)
                vloss = criterion(preds, labels)
                vrunning += vloss.item() * imgs.size(0)

        epoch_val_loss = vrunning / (len(val_loader.dataset))
        val_losses.append(epoch_val_loss)

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # save intermediate model
        torch.save(model.state_dict(), os.path.join(out_dir, f"resnet18_epoch{epoch+1}.pt"))

    # final save
    final_path = os.path.join(out_dir, "resnet18_final.pt")
    torch.save(model.state_dict(), final_path)
    print("Saved final model to:", final_path)

    # plot
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="train")
    plt.plot(range(1, epochs+1), val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    print("Saved loss curve to:", os.path.join(out_dir, "loss_curve.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../data/ava_subset/labels.csv")
    parser.add_argument("--img_dir", type=str, default="../data/ava_subset/images")
    parser.add_argument("--out", type=str, default="../results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args.csv, args.img_dir, args.out, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
