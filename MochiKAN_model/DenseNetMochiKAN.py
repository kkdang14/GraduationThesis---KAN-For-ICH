import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
import math

from version.mochikan.mochikan import MochiKAN 

class DenseNetFastKAN(nn.Module):
    def __init__(self, hidden_dims=None, num_classes=10, pretrained=True, freeze_backbone=True, densenet_version='121'):
        super(DenseNetFastKAN, self).__init__()
        
        # Load pre-trained DenseNet model
        if densenet_version == '121':
            self.densenet = models.densenet121(weights=None if not pretrained else "DEFAULT")
        elif densenet_version == '161':
            self.densenet = models.densenet161(weights=None if not pretrained else "DEFAULT")
        elif densenet_version == '169':
            self.densenet = models.densenet169(pretrained=pretrained)
        elif densenet_version == '201':
            self.densenet = models.densenet201(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported DenseNet version: {densenet_version}")

        # Freeze DenseNet layers if specified
        if freeze_backbone:
            for param in self.densenet.parameters():
                param.requires_grad = False

        # Get the feature dimension from DenseNet classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # Remove the classifier
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Create the complete layers list including input and output dimensions
        layers_hidden = [num_features] + hidden_dims + [num_classes]
        
        # Create FastKAN network with the specified architecture
        self.mochikan = MochiKAN(
            layers_hidden=layers_hidden,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=8,
            use_base_update=True,
            base_activation=nn.SiLU(),
            spline_weight_init_scale=0.1,
            wendland_bandwidth=None,  # Use default bandwidth
            learnable_centers=False,
            learnable_bandwidth=False
        )   

    def forward(self, x):
        x = self.densenet(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.mochikan(x)
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

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = DenseNetFastKAN().to(device)
print(model)    
print_parameter_details(model)
count_model_size(model)
print(f"Model size: {count_model_size(model):.2f} MB")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model)}")

# Clean up
print("\nCleaning up...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
print("Done! DenseNet + FastKAN implementation complete.")