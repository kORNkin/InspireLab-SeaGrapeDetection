"""
Step 4: Train the stage-2 maturity classifier on padded ground-truth crops.

Two decisions carry most of the accuracy here.

Colour augmentation is nearly disabled. The class *is* the colour --
Darkening, Harvestable and Whitening differ by little else -- so saturation and
hue jitter do not regularise the model, they relabel the data. The existing
train_classifier.py used saturation=0.2, which is enough to push a Harvestable
crop across the Darkening boundary. Brightness and contrast are kept for
underwater lighting; hue and saturation are held near zero. All the real
augmentation budget goes to geometry, which cannot change a maturity label.

Imbalance is handled by resampling, not by loss weights. Harvestable outnumbers
Whitening 13:1. WeightedRandomSampler equalises what the model sees per epoch;
applying class weights to the loss on top of that would double-count the
correction and push the model into over-predicting Whitening.

Model selection uses macro recall, not accuracy. Always guessing Harvestable
scores 76% accuracy on this distribution and is worthless.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
CROPS = BASE_DIR / "build" / "crops"
OUT = BASE_DIR / "build" / "runs" / "stage2_classifier"

IMG_SIZE = 128
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Geometry: a grape has no canonical orientation, so this is free data.
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        # Photometric: brightness/contrast only. hue and saturation stay near zero
        # because they move crops across the maturity boundary.
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.05, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return train_tf, eval_tf


def macro_recall(conf):
    """Mean per-class recall from a confusion matrix (rows = true)."""
    per = [conf[i, i] / conf[i].sum() if conf[i].sum() else 0.0 for i in range(len(conf))]
    return float(np.mean(per)), per


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    conf = np.zeros((n_classes, n_classes), int)
    for images, labels in loader:
        preds = model(images.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(labels.numpy(), preds):
            conf[t, p] += 1
    return conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    train_tf, eval_tf = build_transforms()

    train_ds = datasets.ImageFolder(CROPS / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(CROPS / "valid", transform=eval_tf)
    test_ds = datasets.ImageFolder(CROPS / "test", transform=eval_tf)
    class_names = train_ds.classes
    n_classes = len(class_names)

    counts = Counter(y for _, y in train_ds.samples)
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    for i, name in enumerate(class_names):
        print(f"  {name:<12} train={counts[i]:>5}")

    # Equalise class exposure per epoch. Weight is per-sample = 1/count(its class).
    sample_weights = [1.0 / counts[y] for _, y in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=6, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, num_workers=6)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=6)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    model = model.to(device)

    # No class weights here: the sampler already balances the classes.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = OUT / "classifier_mobilenet_v3.pth"
    best_recall, stale = -1.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()

        conf = evaluate(model, valid_loader, device, n_classes)
        recall, per_class = macro_recall(conf)
        acc = np.trace(conf) / conf.sum()
        history.append({"epoch": epoch, "train_loss": loss_sum / seen,
                        "val_acc": float(acc), "val_macro_recall": recall})

        per_str = " ".join(f"{n[:4]}={r:.3f}" for n, r in zip(class_names, per_class))
        print(f"Epoch {epoch:03d}/{args.epochs} | loss {loss_sum / seen:.4f} | "
              f"val acc {acc:.4f} | macro recall {recall:.4f} | {per_str}")

        if recall > best_recall:
            best_recall, stale = recall, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_macro_recall": recall,
                "img_size": IMG_SIZE,
                "context_pad": 0.15,  # inference must crop with the same padding
            }, ckpt_path)
            print(f"  -> saved (macro recall {recall:.4f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"\nEarly stop at epoch {epoch}; no gain for {args.patience} epochs.")
                break

    print(f"\nBest val macro recall: {best_recall:.4f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    conf = evaluate(model, test_loader, device, n_classes)
    recall, per_class = macro_recall(conf)

    print("\n=== Test confusion matrix (rows = truth, cols = predicted) ===")
    print(f"{'':<14}" + "".join(f"{n[:11]:>12}" for n in class_names) + f"{'recall':>10}")
    for i, name in enumerate(class_names):
        row = "".join(f"{conf[i, j]:>12}" for j in range(n_classes))
        print(f"{name:<14}{row}{per_class[i]:>10.3f}")
    print(f"\nTest accuracy      : {np.trace(conf) / conf.sum():.4f}")
    print(f"Test macro recall  : {recall:.4f}   <- the number that matters")

    (OUT / "results.json").write_text(json.dumps({
        "class_names": class_names,
        "best_val_macro_recall": best_recall,
        "test_accuracy": float(np.trace(conf) / conf.sum()),
        "test_macro_recall": recall,
        "test_per_class_recall": dict(zip(class_names, per_class)),
        "test_confusion_matrix": conf.tolist(),
        "history": history,
    }, indent=2))
    print(f"\nWrote {ckpt_path}\nWrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
