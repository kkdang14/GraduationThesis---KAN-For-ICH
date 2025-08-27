import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc

from version.fastkan.fastkan import FastKAN  # Assuming FastKAN is defined in fastkan.py

class MobileNetV2FastKAN(nn.Module):
    def __init__(self, hidden_dims=None, num_classes=20, pretrained=True, freeze_backbone=True):
        super(MobileNetV2FastKAN, self).__init__()
        
        # Load pre-trained mobilenet model
        self.mobilenet = models.mobilenet_v2(weights=None if not pretrained else "DEFAULT")

        # Freeze MobileNet layers if specified
        if freeze_backbone:
            for param in self.mobilenet.parameters():
                param.requires_grad = False

        # Get the feature dimension from mobilenet
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Identity()  # Remove the classifier
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [768, 384]
        
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
        x = self.mobilenet(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fastkan(x)
        return x

def print_parameter_details(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()  # Number of elements in the tensor
            total_params += params
            print(f"{name}: {params}")
    print(f"Total trainable parameters: {total_params}")
    

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV2FastKAN().to(device)
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
    
print("Done! MobileNet + FastKAN implementation complete.")