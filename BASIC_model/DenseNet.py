import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import gc
import matplotlib.pyplot as plt
from torchvision import models

class DenseNet(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(DenseNet, self).__init__()
        
        # Use DenseNet121 from torchvision
        self.densenet = models.densenet121(weights='DEFAULT' if pretrained else None)
        
        # Replace the classifier head for our number of classes
        # DenseNet121 has 1024 features before the classifier
        self.densenet.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.densenet(x)

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
print("DENSENET MODEL")
print("=" * 80)

model = DenseNet(num_classes=20, pretrained=False).to(device)
print("Model Architecture:")
print("=" * 80)
print_parameter_details(model)
print(f"Model size: {count_model_size(model):.2f} MB")  
print(f"Total trainable parameters: {count_parameters(model):,}")