import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    """
    Implements Partial Cross-Entropy (PCE) for weakly supervised learning.
    Leverages PyTorch's optimized C++ backend to natively drop unannotated background 
    pixels from the computational graph, computing gradients only for sparse labels.
    """
    def __init__(self, ignore_index=-1):
        super(PartialCrossEntropyLoss, self).__init__()
        # Explicitly passing ignore_index bypasses manual masking, saving VRAM.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

class PartialFocalLoss(nn.Module):
    """
    Partial Focal Loss designed to combat severe class imbalance in remote sensing 
    (e.g., small buildings vs. vast background forests). Restricts the gradient penalty 
    purely to annotated points via the normalization mask: Σ(Focal loss * MASK) / Σ(MASK).
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
        super(PartialFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Can be a tensor of weights per class
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # 1. Calculate standard CE loss without reduction (per-pixel)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        # 2. Calculate pt (probability of the correct class)
        pt = torch.exp(-ce_loss)
        
        # 3. Apply the Focal Loss formula: (1-pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Create a mask of pixels that are NOT the ignore_index
        mask = (targets != self.ignore_index).float()
        
        # 5. Sum the loss and divide by the number of labeled points 
        # This matches the Σ(Loss * MASK) / Σ(MASK) requirement
        loss = focal_loss.sum() / mask.sum().clamp(min=1)
        
        return loss

def get_loss_function(loss_type='pce', ignore_index=-1):
    """
    Loss factory to easily swap between objective functions during hyperparameter tuning.
    """
    if loss_type == 'pce':
        return PartialCrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == 'focal':
        return PartialFocalLoss(ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")