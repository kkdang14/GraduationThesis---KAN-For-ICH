import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchsummary import summary
import gc
import matplotlib.pyplot as plt

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=False):
        super(MobileNetV2, self).__init__()
        
        # Load MobileNet model (MobileNetV2 is more common and stable)
        self.mobilenet = models.mobilenet_v2(weights=None if not pretrained else "DEFAULT")
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.mobilenet.parameters():
                param.requires_grad = False
        
        # Get the number of features from the original classifier
        num_features = self.mobilenet.classifier[1].in_features
        
        # Replace the classifier head for our number of classes
        self.mobilenet.classifier = nn.Linear(num_features, num_classes)
        
        # If backbone is frozen, make sure new classifier is trainable
        if freeze_backbone:
            for param in self.mobilenet.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.mobilenet(x)

def print_parameter_details(model):
    total_params = 0
    trainable_params = 0
    
    print("Layer-wise parameter count:")
    print("-" * 60)
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        
        if parameter.requires_grad:
            trainable_params += params
            print(f"{name}: {params:,} (trainable)")
        else:
            print(f"{name}: {params:,} (frozen)")
    
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model with different configurations
print("=" * 80)
print("MOBILENETV2 MODEL")
print("=" * 80)

model = MobileNetV2(num_classes=20, pretrained=False).to(device)
print("Model Architecture:")
print("=" * 80)
print_parameter_details(model)
print(f"Model size: {count_model_size(model):.2f} MB")  
print(f"Total trainable parameters: {count_parameters(model):,}")