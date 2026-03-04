import json
import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb

from cnnmodels import ResNN

TRAIN_DIR = "~/datasets/hotdog-nothotdog/train"
VAL_DIR   = "~/datasets/hotdog-nothotdog/test"

IMG_SIZE = 192
BATCH_SIZE = 128
EPOCHS = 80
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    return (preds.view(-1) == y).float().mean().item()


def best_threshold_and_acc(model, loader, device):
    model.eval()
    probs_list, ys_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            probs_list.append(p)
            ys_list.append(y.numpy().reshape(-1))

    probs = np.concatenate(probs_list)
    ys = np.concatenate(ys_list).astype(np.int32)

    best_acc, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 37):
        acc = ((probs >= t).astype(np.int32) == ys).mean()
        if acc > best_acc:
            best_acc, best_t = float(acc), float(t)
    return best_t, best_acc


def main():
    wandb.init(
        project="hotdog-classifier",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "img_size": IMG_SIZE,
            "model": "ResNN + CosineAnnealingLR + RandomErasing",
        }
    )

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tfms)

    from collections import Counter
    print("train counts:", Counter([y for _, y in train_ds.samples]))
    print("val counts:", Counter([y for _, y in val_ds.samples]))

    class_names = train_ds.classes
    print("classes:", class_names)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = ResNN().to(device)

    print("device:", device)
    print("model param device:", next(model.parameters()).device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    os.makedirs("model", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1, 1)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # torch.cuda.synchronize()

            train_loss += loss.item()
            train_acc += accuracy_from_logits(logits.detach(), y.long().view(-1))

        train_loss /= len(train_loader)
        train_acc  /= len(train_loader)

        model.eval()
        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).float().view(-1, 1)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_acc += accuracy_from_logits(logits, y.long().view(-1))

        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)

        best_t, best_acc = best_threshold_and_acc(model, val_loader, device)
        wandb.log({"val_best_acc": best_acc, "val_best_threshold": best_t})
        print(f"val_best_acc={best_acc:.3f} @ threshold={best_t:.2f}")

        scheduler.step()

        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model/model.pt")
            wandb.save("model/model.pt")
            with open("model/class_names.json", "w", encoding="utf-8") as f:
                json.dump(class_names, f)
            print(f"  ✅ saved new best: val_acc={best_val_acc:.3f}")

    print("Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
