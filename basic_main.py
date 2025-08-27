import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json
import time
import argparse
import zipfile
from PIL import Image
import io
from torch.utils.data import Dataset
import logging


from BASIC_model import BasicCNN, ResNet, ConvNeXt, EfficientNetV2, DenseNet, MobileNetV2

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class CheckpointManager:
    def __init__(self, checkpoint_dir, model_name="model", keep_last_n=5, log_file=None):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Initialize logging
        self.logger = logging.getLogger(f"CheckpointManager_{model_name}")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.addHandler(logging.NullHandler())
        
    def save_checkpoint(self, epoch, model, optimizer, scheduler, train_losses, val_losses, 
                        train_accs, val_accs, best_val_loss, total_time, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_loss': best_val_loss,
            'training_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_checkpoint_epoch{epoch}.pth')
        self._remove_old_checkpoints()
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint of {self.model_name} model at epoch {epoch} to {checkpoint_path}")
        print(f"✓ Saved checkpoint of {self.model_name} model at epoch {epoch} to {checkpoint_path}")
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best {self.model_name} model at epoch {epoch} with val_loss: {best_val_loss:.4f}")
            print(f"✓ Saved best {self.model_name} model at epoch {epoch} with val_loss: {best_val_loss:.4f}")
        return checkpoint_path
    
    def _remove_old_checkpoints(self):
        checkpoint_files = sorted(
            [f for f in os.listdir(self.checkpoint_dir) 
            if f.startswith(f'{self.model_name}_checkpoint_epoch') and f.endswith('.pth')],
            key=lambda x: int(x.split('epoch')[1].split('.pth')[0])
        )
        if len(checkpoint_files) > self.keep_last_n:
            for file in checkpoint_files[:-self.keep_last_n]:
                file_path = os.path.join(self.checkpoint_dir, file)
                try:
                    os.remove(file_path)
                    self.logger.info(f"Removed old checkpoint: {file_path}")
                    print(f"Removed old checkpoint: {file_path}")
                except OSError as e:
                    error_msg = f"Error removing old checkpoint {file_path}: {e}"
                    self.logger.error(error_msg)
                    print(error_msg)
                    # Raise exception for critical errors (e.g., permission denied)
                    if isinstance(e, (PermissionError, IOError)):
                        raise OSError(f"Critical error removing checkpoint {file_path}: {e}")
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return {
            'epoch': checkpoint['epoch'],
            'train_losses': checkpoint['train_losses'],
            'val_losses': checkpoint['val_losses'],
            'train_accs': checkpoint['train_accs'],
            'val_accs': checkpoint['val_accs'],
            'best_val_loss': checkpoint['best_val_loss'],
            'training_time': checkpoint.get('training_time', 0.0)
        }
    
    def get_latest_checkpoint(self):
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir) 
            if f.startswith(f'{self.model_name}_checkpoint_epoch') and f.endswith('.pth')
        ]
        if not checkpoint_files:
            return None
        latest_file = max(
            checkpoint_files,
            key=lambda x: int(x.split('epoch')[1].split('.pth')[0])
        )
        return os.path.join(self.checkpoint_dir, latest_file)
    
    
def train(model, train_loader, criterion, optimizer, device, epoch, config, model_name):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_loader)
    num_epochs = int(config['num_epochs'])

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        batch_progress = (batch_idx + 1) / num_batches * 100
        print(f"\rEpoch {epoch+1}/{num_epochs} of {model_name} model - Batch {batch_idx+1}/{num_batches} - Train Running Loss: {running_loss/total:.4f}, Train Running Acc: {correct/total:.4f} - Progress: [{batch_progress:.1f}%]", end='', flush=True)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    epoch_progress = (epoch + 1) / num_epochs * 100
    print(f"\rEpoch {epoch+1}/{num_epochs} of {model_name} model - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - Progress: [{epoch_progress:.1f}%]")
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, epoch, config, model_name):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(val_loader)
    num_epochs = int(config['num_epochs'])

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            batch_progress = (batch_idx + 1) / num_batches * 100
            print(f"\rEpoch {epoch+1}/{num_epochs} of {model_name} model - Batch {batch_idx+1}/{num_batches} - Val Running Loss: {running_loss/total:.4f}, Val Running Acc: {correct/total:.4f} - Progress: [{batch_progress:.1f}%]", end='', flush=True)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total
    epoch_progress = (epoch + 1) / num_epochs * 100
    print(f"\rEpoch {epoch+1}/{num_epochs} of {model_name} model - Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f} - Progress [{epoch_progress:.1f}%]")
    return epoch_loss, epoch_acc

def save_training_config(config_path, config):
    """Save config to JSON, excluding non-serializable objects."""
    # Create a copy of config without the model classes
    serializable_config = config.copy()
    
    # Convert models list to only include names (remove class objects)
    if 'models' in serializable_config:
        serializable_config['models'] = [{'name': m['name']} for m in serializable_config['models']]
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, np.array(val_losses) - np.array(train_losses), 'g-', label='Overfitting Gap')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss - Train Loss')
    plt.legend()
    plt.title('Overfitting Monitor')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model_name, model_class, config, device, train_loader, val_loader, num_classes, use_dataparallel=True):
    model = model_class(num_classes=num_classes).to(device)
    if use_dataparallel:
        model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler_factor'], patience=config['scheduler_patience'])
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'], min_delta=config['early_stopping_min_delta']
    )
    checkpoint_manager = CheckpointManager(config['checkpoint_dir'], model_name, 
                                        keep_last_n=3, log_file=os.path.join(config['results_dir'], f'{model_name}_training.log'))
    
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    accumulated_training_time = 0.0  # Initialize to avoid UnboundLocalError
    epoch_stopping = config['num_epochs']
    
    if latest_checkpoint:
        print(f"\nFound existing checkpoint for {model_name}: {latest_checkpoint}")
        checkpoint_data = checkpoint_manager.load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
        start_epoch = checkpoint_data['epoch'] + 1
        best_val_loss = checkpoint_data['best_val_loss']
        train_losses = checkpoint_data['train_losses']
        val_losses = checkpoint_data['val_losses']
        train_accs = checkpoint_data['train_accs']
        val_accs = checkpoint_data['val_accs']
        accumulated_training_time = checkpoint_data.get('training_time', 0.0)
        print(f"Resumed {model_name} from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    
    print(f"\nStarting training {model_name} from epoch {start_epoch + 1}")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f'\n=== {model_name} Epoch {epoch+1}/{config["num_epochs"]} ===')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, config, model_name)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, config, model_name)
        print()  # New line after epoch progress
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % config['checkpoint_every'] == 0 or is_best:
            print(f"\nSaving checkpoint for {model_name} at epoch {epoch + 1}")
            epoch_time = time.time() - start_time
            total_time = accumulated_training_time + epoch_time
            checkpoint_manager.save_checkpoint(
                epoch, model, optimizer, scheduler, train_losses, val_losses, train_accs, val_accs, best_val_loss, total_time, is_best)
        
        if early_stopping(val_loss, model):
            epoch_stopping = epoch + 1
            print(f"\nEarly stopping triggered for {model_name} at epoch {epoch}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break
        
    current_session_time = time.time() - start_time
    total_time = accumulated_training_time + current_session_time
    print(f"\nTraining {model_name} completed in {total_time:.2f} seconds")
    
    checkpoint_manager.save_checkpoint(
        epoch, model, optimizer, scheduler, train_losses, val_losses, train_accs, val_accs, best_val_loss, total_time, is_best=True )
    
    plot_path = os.path.join(config['results_dir'], f'{model_name}_training_history.png')
    print(f"{model_name} training history was saved")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    final_metrics = {
        'model_name': model_name,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
        'early_stopped': len(train_losses) < config['num_epochs'],
        'epoch_stopping': epoch_stopping,
        'training_time': total_time,
    }
    
    metrics_path = os.path.join(config['results_dir'], f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n=== {model_name} Training Complete ===")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"Results saved to: {config['results_dir']}")

def get_model_registry():
    """Return a dictionary mapping model names to their classes."""
    return {
        'resnet': ResNet,
        'basic_cnn': BasicCNN,
        'convnext': ConvNeXt,
        'densenet': DenseNet,
        'mobilenetv2': MobileNetV2,
        'efficientnetv2': EfficientNetV2
    }
    
class ZipImageDataset(Dataset):
    """Custom Dataset to load images directly from a zip file."""
    def __init__(self, zip_path, subset='train', transform=None):
        self.zip_path = zip_path
        self.subset = subset  # 'train' or 'val'
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.classes = []

        # Supported image extensions
        self.valid_extensions = ('.jpg', '.jpeg', '.png')

        # Open zip file and collect image paths and labels
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List all files in the zip
            all_files = zf.namelist()
            # Filter files for the specified subset (train or val)
            subset_prefix = f'data/{subset}/'
            image_files = [
                f for f in all_files
                if f.startswith(subset_prefix) and f.lower().endswith(self.valid_extensions)
            ]

            # Build class index and collect image paths
            classes = set()
            for file_path in image_files:
                # Extract class name from path (e.g., data/train/class1/image.jpg -> class1)
                class_name = file_path.split('/')[2]
                classes.add(class_name)
            self.classes = sorted(list(classes))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            # Store image paths and corresponding labels
            for file_path in image_files:
                class_name = file_path.split('/')[2]
                self.image_paths.append(file_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Open zip file and read image
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.image_paths[idx]) as f:
                img_data = f.read()
                img = Image.open(io.BytesIO(img_data)).convert('RGB')

        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

def main():
    parser = argparse.ArgumentParser(description="Train multiple KAN base models")
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--model', type=str, help='Specific model to train (optional)')
    parser.add_argument('--data_zip', type=str, default='data.zip', help='Path to data zip file')
    args = parser.parse_args()
    
    # Get model registry
    model_registry = get_model_registry()
    
    # Default configuration - store models as name-class pairs
    config = {
        'batch_size': 128,  # KAN models might need smaller batch sizes due to memory requirements
        'learning_rate': 0.001,  # Slightly lower learning rate for KAN models
        'num_epochs': 75,
        'early_stopping_patience': 10,  # Increased patience for KAN models
        'early_stopping_min_delta': 0.001,
        'scheduler_patience': 5,  # Increased patience for learning rate scheduler
        'scheduler_factor': 0.5,
        'checkpoint_every': 5,
        'checkpoint_dir': 'basic_checkpoints',
        'results_dir': 'basic_training_result',
        'models': [
            {'name': 'resnet', 'class': ResNet},
            {'name': 'basic_cnn', 'class': BasicCNN},
            {'name': 'convnext', 'class': ConvNeXt},
            {'name': 'densenet', 'class': DenseNet},
            {'name': 'mobilenet', 'class': MobileNetV2},
            {'name': 'efficientnetv2', 'class': EfficientNetV2}
        ]
    }
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
            
            # Reconstruct model classes from names if loading from JSON
            if 'models' in loaded_config:
                config['models'] = []
                for model_info in loaded_config['models']:
                    model_name = model_info['name']
                    if model_name in model_registry:
                        config['models'].append({
                            'name': model_name,
                            'class': model_registry[model_name]
                        })
    
    # Filter models if specific model is requested
    if args.model:
        config['models'] = [m for m in config['models'] if m['name'] == args.model]
        if not config['models']:
            raise ValueError(f"Model {args.model} not found in configuration")
        
    # Verify data.zip exists
    if not os.path.exists(args.data_zip):
        raise FileNotFoundError(f"Data zip file not found at {args.data_zip}")
    
    # Set up device (use all available GPUs if present)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"Using {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # If multiple GPUs are available, use DataParallel for model training
    use_dataparallel = torch.cuda.is_available() and num_gpus > 1
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    model_names = [m['name'] for m in config['models']]
    
    # Save configuration (this will now work without the JSON serialization error)
    config_path = os.path.join(config['results_dir'], f'{"_".join(model_names)}_training_config.json')
    save_training_config(config_path, config)
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets using ZipImageDataset
    train_dataset = ZipImageDataset(zip_path=args.data_zip, subset='train', transform=train_transform)
    val_dataset = ZipImageDataset(zip_path=args.data_zip, subset='val', transform=val_transform)

    # Verify datasets are not empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"No images found in {args.data_zip} for {'train' if len(train_dataset) == 0 else 'val'} subset")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Dataset classes: {class_names} (total: {num_classes})")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Train each model
    for model_config in config['models']:
        model_name = model_config['name']
        model_class = model_config['class']
        print(f"\nTraining model: {model_name}")
        train_model(model_name, model_class, config, device, train_loader, val_loader, num_classes, use_dataparallel)

if __name__ == "__main__":
    main()