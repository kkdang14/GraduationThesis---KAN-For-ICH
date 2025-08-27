import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, num_classes=20, pretrained=True, freeze_backbone=True):
        super(VGG16, self).__init__()
        
        # Use ConvNeXt-Base from torchvision
        self.vgg16 = models.vgg16(weights='DEFAULT' if pretrained else None)
        
        if freeze_backbone:
            # Freeze VGG16 layers if specified
            for param in self.vgg16.parameters():
                param.requires_grad = False
        
        # Get the number of input features for the classifier
        num_features = self.vgg16.classifier[6].in_features
        
        # Replace the final linear layer
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.vgg16(x)

def print_parameter_details(model):
    """Print layer-wise parameter counts and trainability."""
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
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_model_size(model):
    """Calculate model size in MB (assuming float32 parameters)."""
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

# Create model
print("=" * 80)
print("CONVNEXT MODEL")
print("=" * 80)

model = VGG16(num_classes=20, pretrained=False).to(device)
print("Model Architecture:")
print("=" * 80)
print_parameter_details(model)
print(f"Model size: {count_model_size(model):.2f} MB")  
print(f"Total trainable parameters: {count_parameters(model):,}")