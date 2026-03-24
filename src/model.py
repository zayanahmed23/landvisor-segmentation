import segmentation_models_pytorch as smp
import torch.nn as nn

def get_model(model_name='resnet34', num_classes=7):
    """
    Instantiates a U-Net architecture with an ImageNet-pretrained encoder.
    Utilizing transfer learning here is critical; the pre-trained spatial hierarchy 
    compensates for the lack of dense annotations in our weakly supervised setting.
    """
    
    # We use ResNet34 as it is lightweight and powerful for assessments
    model = smp.Unet(
        encoder_name=model_name,        
        encoder_weights="imagenet",     # use pre-trained weights from ImageNet
        in_channels=3,                  # model input channels (RGB)
        classes=num_classes,            # model output channels (number of classes)
        activation=None                 # We keep None because our Loss function 
                                        # (CrossEntropy) applies Softmax internally
    )
    
    return model

class SegmentationModel(nn.Module):
    """
    Module wrapper for the segmentation network. 
    Abstracts the underlying SMP implementation to ensure the main training loop 
    remains completely decoupled from the specific choice of model library.
    """
    def __init__(self, model_name='resnet34', num_classes=7):
        super(SegmentationModel, self).__init__()
        self.model = get_model(model_name, num_classes)

    def forward(self, x):
        return self.model(x)