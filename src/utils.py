import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_iou(preds, targets, num_classes, ignore_index=-1):
    """
    Calculates Intersection over Union (IoU) for each class.
    Professional implementation using a confusion matrix approach.
    """
    ious = []
    # Flatten tensors for easier calculation
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        # Create masks for the current class
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        
        # Intersection: pixels where both are the class
        intersection = (pred_inds & target_inds).sum().item()
        
        # Union: pixels where either is the class (excluding ignore areas)
        # Note: ignore_index pixels are already excluded because targets == cls 
        # won't be true for the ignore_index.
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            ious.append(float('nan'))  # If class isn't present, ignore it
        else:
            ious.append(intersection / union)
            
    return np.nanmean(ious)  # Returns mean IoU across all present classes

def visualize_results(image, sparse_mask, prediction, full_gt, save_path):
    """
    Generates a 4-panel figure for the Technical Report.
    Shows the input, the sparse points (the challenge), 
    the model output, and the real ground truth.
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original Image (Convert from Tensor CHW to HWC)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    ax[0].imshow(img_np)
    ax[0].set_title("Input Image")
    
    # 2. Sparse Points (The simulated "Incomplete Tagging")
    # We mask out the ignore_index so they appear as dots
    sparse_np = sparse_mask.cpu().numpy()
    ax[1].imshow(img_np, alpha=0.5) # Background image
    ax[1].imshow(np.where(sparse_np == -1, np.nan, sparse_np), cmap='jet', interpolation='none')
    ax[1].set_title("Simulated Point Labels")
    
    # 3. Model Prediction
    pred_np = prediction.cpu().numpy()
    ax[2].imshow(pred_np, cmap='jet')
    ax[2].set_title("Model Prediction")
    
    # 4. Full Ground Truth (The "Goal")
    gt_np = full_gt.cpu().numpy()
    ax[3].imshow(gt_np, cmap='jet')
    ax[3].set_title("Full Ground Truth")
    
    for a in ax:
        a.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()