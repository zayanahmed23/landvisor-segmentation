import torch
from torch.utils.data import DataLoader
from src.dataset import LoveDASparseDataset
from src.losses import PartialCrossEntropyLoss
from src.model import SegmentationModel
import torch.optim as optim
import pandas as pd

# We import the core functions and paths directly from your working train.py script
from train import (
    train_one_epoch, validate, 
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, 
    BATCH_SIZE, LR, IGNORE_INDEX, device
)

def run_ablation_study():
    # CPU Hack: Keep epochs very low for the experiment loop
    epochs_per_run = 3 
    
    # The variable being tested: How many points are needed to get a decent mask?
    point_settings = [1, 5, 10, 50]
    results = []

    print(f"Loading static validation set...")
    val_ds = LoveDASparseDataset(img_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, mode='val')
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    for n in point_settings:
        print(f"\n==================================================")
        print(f"  STARTING EXPERIMENT: {n} POINT(S) PER CLASS")
        print(f"==================================================")

        # 1. Initialize training data with the specific point count
        train_ds = LoveDASparseDataset(
            img_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR,
            num_points=n, transform=True, mode='train'
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        # 2. Spin up a fresh, untrained model and optimizer
        model = SegmentationModel(num_classes=7).to(device)
        criterion = PartialCrossEntropyLoss(ignore_index=IGNORE_INDEX)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_iou = 0
        for epoch in range(epochs_per_run):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            mIoU = validate(model, val_loader)
            
            print(f"  [Epoch {epoch+1}/{epochs_per_run}] Loss: {train_loss:.4f} | mIoU: {mIoU:.4f}")

            if mIoU > best_iou:
                best_iou = mIoU

        # 3. Save the best result for this point setting
        results.append({
            "Points_Per_Class": n,
            "Best_mIoU": round(best_iou, 4)
        })

    # 4. Export the final table
    print("\nSaving results to CSV...")
    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    print("Experiments complete! Open experiment_results.csv to see the data for your report.")

if __name__ == "__main__":
    run_ablation_study()