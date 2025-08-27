import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc


from version.kan.kan import KANLayer
from sklearn.metrics import classification_report, confusion_matrix

class DenseNetKAN(nn.Module):
    def __init__(self, hidden_dims=None, num_classes=11, pretrained=True, freeze_backbone=True, densenet_version='121', kan_num_grids=5, kan_spline_order=3):
        """DenseNet + KANLayer implementation"""
        super(DenseNetKAN, self).__init__()
        
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
        
        # Create KANLayer network
        # First layer: ConvNeXt features -> hidden layer
        self.kan_layer1 = KANLayer(
            in_dim=num_features, 
            out_dim=256, 
            num=kan_num_grids,           # number of grid intervals
            k=kan_spline_order,             # spline order
            noise_scale=0.1, 
            scale_base_mu=0.0,
            scale_base_sigma=1.0,
            scale_sp=1.0,
            base_fun=torch.nn.SiLU(),
            grid_eps=0.02,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            device='cpu'  # Will be moved to correct device later
        )
        
        # Second layer: hidden layer -> output classes
        self.kan_layer2 = KANLayer(
            in_dim=256,
            out_dim=num_classes,
            num=5,
            k=3,
            noise_scale=0.1,
            scale_base_mu=0.0,
            scale_base_sigma=1.0,
            scale_sp=1.0,
            base_fun=torch.nn.SiLU(),
            grid_eps=0.02,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            device='cpu'
        )

    def forward(self, x):
        """Forward pass through the network"""
        # Extract features using ConvNeXt
        x = self.convnext(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # Pass through KANLayers
        # KANLayer.forward returns (y, preacts, postacts, postspline)
        # We only need y (the main output)
        output_kan1 = self.kan_layer1(x)
        x = output_kan1[0]
        output_kan2 = self.kan_layer2(x)
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
model = DenseNetKAN().to(device)
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
    
print("Done! DenseNet + Regular KAN implementation complete.")