import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from version.fastkan.fastkan import FastKAN  # Assuming FastKAN is defined in fastkan.py
from sklearn.metrics import classification_report, confusion_matrix

class BasicCNNFastKAN(nn.Module):
    def __init__(self, hidden_dims=None, num_classes=20, freeze_backbone=False):
        super(BasicCNNFastKAN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # 224x224 -> 56x56
            nn.Dropout2d(0.25),
            
            # Second conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # 56x56 -> 14x14
            nn.Dropout2d(0.25),
            
            # Third conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # 14x14 -> 3x3 (with some rounding)
            nn.Dropout2d(0.25),
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze CNN layers if specified
        if freeze_backbone:
            for param in self.conv_layers.parameters():
                param.requires_grad = False
        
        # Feature dimension after global average pooling
        num_features = 256
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Create the complete layers list including input and output dimensions
        layers_hidden = [num_features] + hidden_dims + [num_classes]
        
        # Create FastKAN network with the specified architecture
        self.fastkan = FastKAN(
            layers_hidden=layers_hidden,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=8,
            use_base_update=True
        )

    def forward(self, x):
        # Convolutional feature extraction
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 256]
        
        # FastKAN classification
        x = self.fastkan(x)
        
        return x

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
print("BASIC CNN + FASTKAN MODEL")
print("=" * 80)

model = BasicCNNFastKAN(num_classes=20, freeze_backbone=False).to(device)
print("Model Architecture:")
print("=" * 80)
print_parameter_details(model)
print(f"Model size: {count_model_size(model):.2f} MB")  
print(f"Total trainable parameters: {count_parameters(model):,}")