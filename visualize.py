import torch
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import LoveDASparseDataset
from src.model import SegmentationModel

# Import paths from your existing train script
from train import VAL_IMG_DIR, VAL_MASK_DIR, IGNORE_INDEX

def generate_visual_proof():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    # 1. Load the Model and the saved weights
    model = SegmentationModel(num_classes=7).to(device)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Error: best_model.pth not found. Wait for train.py to finish!")
        return

    # 2. Grab ONE validation image
    val_ds = LoveDASparseDataset(img_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, num_points=10, mode='val')
    image, sparse_mask, full_mask = val_ds[0] # Grab the very first image

    # 3. Make a prediction
    with torch.no_grad():
        img_batch = image.unsqueeze(0).to(device) # Add batch dimension
        output = model(img_batch)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # 4. Plotting the 4-Panel Image
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Standardize image for plotting (C, H, W) -> (H, W, C)
    img_display = image.permute(1, 2, 0).numpy()
    
    axs[0].imshow(img_display)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    # For sparse mask, we only want to show the labeled points, not the background (-1)
    sparse_display = sparse_mask.numpy().astype(float)
    sparse_display[sparse_display == IGNORE_INDEX] = np.nan 
    axs[1].imshow(img_display) # Show image as background
    axs[1].imshow(sparse_display, cmap='jet', alpha=0.7) # Overlay points
    axs[1].set_title("Simulated Points (Input)")
    axs[1].axis('off')

    axs[2].imshow(pred_mask, cmap='jet')
    axs[2].set_title("Model Prediction")
    axs[2].axis('off')

    axs[3].imshow(full_mask.numpy(), cmap='jet')
    axs[3].set_title("Actual Ground Truth")
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig("visual_proof.png", bbox_inches='tight')
    print("Success! Open visual_proof.png to see your results.")

if __name__ == "__main__":
    generate_visual_proof()