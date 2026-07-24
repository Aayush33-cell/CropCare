"""
CropSense — Crop Detection using Transfer Learning (ResNet-50)
=============================================================
PURPOSE  : Image classification of crop types from photos
MODEL    : ResNet-50 pretrained on ImageNet, fine-tuned on crop images
CLASSES  : Jute, Maize, Rice, Sugarcane, Wheat
AUTHOR   : Aayush — University Project Prototype

NOTE: This is the TRAINING script. Run it on Google Colab or a GPU machine.
      For deployment, use app.py with ONNX Runtime instead.
"""

# pip install torch torchvision matplotlib scikit-learn pillow tqdm
import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "data_dir"   : "crop_dataset",   # folder structure explained below
    "num_classes": 5,                 # one per crop type
    "batch_size" : 32,
    "num_epochs" : 20,
    "lr"         : 1e-4,              # learning rate
    "weight_decay": 1e-4,
    "dropout"    : 0.5,
    "img_size"   : 224,               # ResNet expects 224×224
    "val_split"  : 0.15,              # 15 % for validation
    "test_split" : 0.10,              # 10 % for test
    "save_path"  : "crop_model_best.pth",
    "seed"       : 42,
}

CLASS_NAMES = ["jute", "maize", "rice", "sugarcane", "wheat"]

# ══════════════════════════════════════════════════════════════════════════════
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# instead of always the same photo.
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"] + 32, CONFIG["img_size"] + 32)),
    transforms.RandomCrop(CONFIG["img_size"]),       # random 224×224 patch
    transforms.RandomHorizontalFlip(p=0.5),          # mirror image 50% of the time
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),           # rotate up to ±15°
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=0.2, hue=0.1
    ),                                               # vary colour slightly
    transforms.ToTensor(),                           # convert PIL → tensor [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                  # ImageNet mean
        std =[0.229, 0.224, 0.225]                   # ImageNet std
    ),                                               # centre around 0
])

eval_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════════════════════════
def load_data(data_dir):
    """Load images from folder, split into train/val/test subsets."""
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

    n       = len(full_dataset)
    n_val   = int(n * CONFIG["val_split"])
    n_test  = int(n * CONFIG["test_split"])
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )

    # Apply the lighter eval transform to validation and test sets
    val_ds.dataset  = copy.deepcopy(full_dataset)
    test_ds.dataset = copy.deepcopy(full_dataset)
    val_ds.dataset.transform  = eval_transforms
    test_ds.dataset.transform = eval_transforms

    print(f"Dataset: {n} images → {n_train} train, {n_val} val, {n_test} test")
    print(f"Classes: {full_dataset.classes}")

    loaders = {
        "train": DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                            shuffle=True,  num_workers=2, pin_memory=True),
        "val"  : DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True),
        "test" : DataLoader(test_ds,  batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True),
    }
    return loaders, full_dataset.classes

# → Dense (num_classes, Softmax)
def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    freeze_backbone=True  → only train the new head (faster, good for small datasets)
    freeze_backbone=False → train everything (better results with large datasets)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the last residual block (layer4) for some fine-tuning
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace the original 1000-class head with our custom head
    in_features = model.fc.in_features  # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Dropout(p=CONFIG["dropout"]),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
        # Note: no Softmax here — CrossEntropyLoss applies it internally
    )

    return model.to(DEVICE)

# in half. This helps the model converge to a better minimum.
def train_model(model, loaders, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())

    print(f"\n{'='*60}")
    print(f"  Training for {CONFIG['num_epochs']} epochs on {DEVICE}")
    print(f"{'='*60}\n")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t_start = time.time()

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_correct = 0
            total = 0

            for images, labels in loaders[phase]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)          # forward pass
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()              # backpropagation
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()             # weight update

                running_loss    += loss.item() * images.size(0)
                running_correct += (preds == labels).sum().item()
                total           += images.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = running_correct / total * 100

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())
                    torch.save(best_weights, CONFIG["save_path"])

        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch:3d}/{CONFIG['num_epochs']} | "
            f"Train Loss: {history['train_loss'][-1]:.4f} "
            f"Acc: {history['train_acc'][-1]:.1f}% | "
            f"Val Loss: {history['val_loss'][-1]:.4f} "
            f"Acc: {history['val_acc'][-1]:.1f}% | "
            f"{elapsed:.1f}s"
        )

    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    model.load_state_dict(best_weights)
    return model, history

# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n── Classification Report ──────────────────────────────────")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved → confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Validation")
    axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Validation")
    axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Training history saved → training_history.png")

# ══════════════════════════════════════════════════════════════════════════════
def predict_image(image_path: str, model_path: str = CONFIG["save_path"]):
    """
    Load a saved model and predict the crop type for a single image.

    Usage:
        result = predict_image("my_crop_photo.jpg")
        print(result)
    """
    # Load model
    model = build_model(CONFIG["num_classes"], freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Preprocess image
    img = Image.open(image_path).convert("RGB")
    tensor = eval_transforms(img).unsqueeze(0).to(DEVICE)  # add batch dim

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx  = np.argmax(probs)
    top_prob = probs[top_idx]

    print(f"\nImage : {image_path}")
    print(f"Predicted : {CLASS_NAMES[top_idx]} ({top_prob*100:.1f}% confidence)\n")
    print("All probabilities:")
    for name, prob in zip(CLASS_NAMES, probs):
        bar = "█" * int(prob * 30)
        print(f"  {name:<10} {bar:<30} {prob*100:5.1f}%")

    return {"crop": CLASS_NAMES[top_idx], "confidence": float(top_prob), "probs": dict(zip(CLASS_NAMES, probs.tolist()))}

def main():
    # Check dataset exists
    if not os.path.isdir(CONFIG["data_dir"]):
        print(f"\n⚠  Dataset folder '{CONFIG['data_dir']}' not found.")
        print("   Create it with subfolders for each crop class.")
        print("   Example: crop_dataset/Rice/, crop_dataset/Wheat/, …\n")
        return

    # Load data
    loaders, class_names = load_data(CONFIG["data_dir"])

    # Build model
    model = build_model(CONFIG["num_classes"], freeze_backbone=True)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # Train
    model, history = train_model(model, loaders)

    # Visualise training
    plot_history(history)

    # Evaluate on test set
    evaluate_model(model, loaders["test"], class_names)

    print("\n✓  Done! Model saved to:", CONFIG["save_path"])
    print("   To predict a new image:  predict_image('your_photo.jpg')")


if __name__ == "__main__":
    main()
