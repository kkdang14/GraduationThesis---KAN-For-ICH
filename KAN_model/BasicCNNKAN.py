import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import gc
import matplotlib.pyplot as plt

from version.kan.kan import KANLayer

class BasicCNNKAN(nn.Module):
    def __init__(self, num_classes=20,  kan_num_grids=5, kan_spline_order=3):
        super(BasicCNNKAN, self).__init__()

        # Simple 3-layer CNN
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
        
        # Global Average Pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # KANLayer for classification - directly using KANLayer
        self.kan1 = KANLayer(
            in_dim=256, 
            out_dim=128, 
            num=kan_num_grids, 
            k=kan_spline_order,
            noise_scale=0.1,
            scale_base_mu=0.0,
            scale_base_sigma=1.0,
            scale_sp=1.0,
            base_fun=torch.nn.SiLU(),
            grid_eps=0.02,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            save_plot_data=False,  # Set to False for efficiency in training
            device=device,
            sparse_init=False
        )
        
        self.dropout = nn.Dropout(0.5)
        
        self.kan2 = KANLayer(
            in_dim=128, 
            out_dim=num_classes, 
            num=kan_num_grids, 
            k=kan_spline_order,
            noise_scale=0.1,
            scale_base_mu=0.0,
            scale_base_sigma=1.0,
            scale_sp=1.0,
            base_fun=torch.nn.SiLU(),
            grid_eps=0.02,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            save_plot_data=False,
            device=device,
            sparse_init=False
        )

    def forward(self, x):
        # Convolutional feature extraction
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 256]
        
        # First KANLayer - returns (y, preacts, postacts, postspline)
        # We only need y (the output)
        output_kan1 = self.kan1(x)
        x = output_kan1[0]
        x = self.dropout(x)
    
        # Second KANLayer
        output_kan2 = self.kan2(x)
        x = output_kan2[0]
        
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
model = BasicCNNKAN().to(device)
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
    
print("Done! BasicCNN + Regular KAN implementation complete.")