"""
Main training pipeline for Weakly Supervised Semantic Segmentation.
Orchestrates the training loop, feeding sparse point annotations to the model 
while evaluating against dense ground-truth masks to measure true generalization.
"""

import torch
from torch.utils.data import DataLoader
from src.dataset import LoveDASparseDataset
from src.losses import PartialCrossEntropyLoss
from src.model import SegmentationModel
from src.utils import calculate_iou
import torch.optim as optim
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# HARDWARE CONSTRAINTS: Capped at 5 epochs and batch size 4 for local CPU execution.
# In a production GPU environment, these would be scaled significantly (e.g., bs=32, epochs=50+).
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4
NUM_POINTS = 10
IGNORE_INDEX = -1

TRAIN_IMG_DIR = "data/Train/Train/Urban/images_png" 
TRAIN_MASK_DIR = "data/Train/Train/Urban/masks_png"
VAL_IMG_DIR = "data/Val/Val/Urban/images_png"
VAL_MASK_DIR = "data/Val/Val/Urban/masks_png"

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, sparse_masks, _ in tqdm(loader, desc="Training", leave=False):
        images, sparse_masks = images.to(device), sparse_masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, sparse_masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    ious = []
    with torch.no_grad():
        for images, _, full_masks in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            batch_iou = calculate_iou(preds.cpu(), full_masks, num_classes=7, ignore_index=IGNORE_INDEX)
            if not np.isnan(batch_iou):
                ious.append(batch_iou)
                    
    return np.mean(ious) if ious else 0.0

if __name__ == "__main__":
    print(f"Device: {device}")
    
    try:
        train_ds = LoveDASparseDataset(
            img_dir=TRAIN_IMG_DIR, 
            mask_dir=TRAIN_MASK_DIR, 
            num_points=NUM_POINTS, 
            transform=True,
            mode='train'
        )
        
        val_ds = LoveDASparseDataset(
            img_dir=VAL_IMG_DIR, 
            mask_dir=VAL_MASK_DIR, 
            mode='val'
        )
        print(f"Loaded {len(train_ds)} train imgs, {len(val_ds)} val imgs.")
    except FileNotFoundError as e:
        print(f"Dataset path error. Check paths in train.py.\nError: {e}")
        exit()
    
    # TODO: Add Weights & Biases (wandb) logging and increase num_workers if moving to GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = SegmentationModel(num_classes=7).to(device)
    criterion = PartialCrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_iou = 0
    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        mIoU = validate(model, val_loader)
        
        print(f"Loss: {train_loss:.4f} | mIoU: {mIoU:.4f}")
        
        if mIoU > best_iou:
            best_iou = mIoU
            torch.save(model.state_dict(), "best_model.pth")
            print("--> Model saved")