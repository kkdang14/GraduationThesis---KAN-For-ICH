import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import argparse
import gc
from sklearn.metrics import classification_report, confusion_matrix
import zipfile
from PIL import Image
import io
from torch.utils.data import Dataset
import time

from BASIC_model import ResNet, BasicCNN, ConvNeXt, DenseNet, EfficientNetV2, MobileNetV2, VGG16

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
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / 1024 / 1024

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_test_config(config_path, config):
    serializable_config = config.copy()
    serializable_config['models'] = [{'name': m['name']} for m in serializable_config['models']]
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=4)

def plot_confusion_matrix(conf_matrix, class_names, model_name, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(model_name, model_class, config, device, test_loader, num_classes):
    model = model_class(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_path = os.path.join(config['checkpoint_dir'], f'{model_name}_best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found for {model_name} at {checkpoint_path}")
    
    model = load_checkpoint(checkpoint_path, model, device)
    model.eval()
    
    print(f"\nTesting KAN-based model: {model_name}")
    print_parameter_details(model)
    print(f"Model size: {count_model_size(model):.2f} MB")
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    num_batches = len(test_loader)
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            batch_progress = (batch_idx + 1) / num_batches * 100
            print(f"\rBatch {batch_idx+1}/{num_batches} - Test Running Loss: {running_loss/total:.4f}, Test Running Acc: {correct/total:.4f} - [{batch_progress:.1f}%]", end='', flush=True)
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    testing_time = time.time() - start_time
    
    print(f"\n\nTesting complete in {testing_time:.2f} seconds.")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}")
    
    class_names = test_loader.dataset.classes
    clf_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'model_name': model_name,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'per_class_metrics': {
            cls: {
                'precision': clf_report[cls]['precision'],
                'recall': clf_report[cls]['recall'],
                'f1_score': clf_report[cls]['f1-score'],
                'support': clf_report[cls]['support']
            } for cls in class_names
        },
        'average_metrics': {
            'macro_precision': clf_report['macro avg']['precision'],
            'macro_recall': clf_report['macro avg']['recall'],
            'macro_f1_score': clf_report['macro avg']['f1-score'],
            'weighted_precision': clf_report['weighted avg']['precision'],
            'weighted_recall': clf_report['weighted avg']['recall'],
            'weighted_f1_score': clf_report['weighted avg']['f1-score']
        },
        'testing_time_seconds': testing_time,
        'confusion_matrix': conf_matrix.tolist(),
        'model_size_mb': count_model_size(model),
        'num_classes': num_classes,
        'class_names': class_names
    }
    
    results_dir = config['results_dir']
    metrics_path = os.path.join(results_dir, f'{model_name}_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    cm_path = os.path.join(results_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(conf_matrix, class_names, model_name, cm_path)
    
    print(f"\n=== {model_name} Test Complete ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nAverage Metrics:")
    print(f"Macro Precision: {metrics['average_metrics']['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['average_metrics']['macro_recall']:.4f}")
    print(f"Macro F1-Score: {metrics['average_metrics']['macro_f1_score']:.4f}")
    print(f"Weighted Precision: {metrics['average_metrics']['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['average_metrics']['weighted_recall']:.4f}")
    print(f"Weighted F1-Score: {metrics['average_metrics']['weighted_f1_score']:.4f}")
    print(f"Results saved to: {results_dir}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics

def get_model_registry():
    return {
        'resnet': ResNet,
        'basic_cnn': BasicCNN,
        'convnext': ConvNeXt,
        'densenet': DenseNet,
        'mobilenet': MobileNetV2,
        'efficientnetv2': EfficientNetV2,
        'vgg16': VGG16
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
    parser = argparse.ArgumentParser(description="Evaluate MLP models on test data")
    parser.add_argument('--config', type=str, help='Path to config JSON file (from training)')
    parser.add_argument('--model', type=str, help='Specific model to test (optional)')
    parser.add_argument('--data_zip', type=str, default='data.zip', help='Path to data zip file')
    args = parser.parse_args()
    
    model_registry = get_model_registry()
    
    config = {
        'batch_size': 32,
        'checkpoint_dir': 'basic_checkpoints',
        'results_dir': 'basic_testing_result',
        'models': [
            {'name': 'resnet', 'class': ResNet},
            {'name': 'basiccnn', 'class': BasicCNN},    
            {'name': 'convnext', 'class': ConvNeXt},
            {'name': 'densenet', 'class': DenseNet},
            {'name': 'mobilenetv2', 'class': MobileNetV2},
            {'name': 'efficientnetv2', 'class': EfficientNetV2},
            {'name': 'vgg16', 'class': VGG16}
        ]
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
            if 'models' in loaded_config:
                config['models'] = []
                for model_info in loaded_config['models']:
                    model_name = model_info['name']
                    if model_name in model_registry:
                        config['models'].append({
                            'name': model_name,
                            'class': model_registry[model_name]
                        })
    
    if args.model:
        config['models'] = [m for m in config['models'] if m['name'] == args.model]
        if not config['models']:
            raise ValueError(f"Model {args.model} not found in configuration")
        
    # Verify data.zip exists
    if not os.path.exists(args.data_zip):
        raise FileNotFoundError(f"Data zip file not found at {args.data_zip}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config['results_dir'], exist_ok=True)
    
    model_name = config['models'][0]['name'] if config['models'] else 'default_model'
    print(f"Testing model: {model_name}")
    
    config_path = os.path.join(config['results_dir'], f'{model_name}_test_config.json')
    save_test_config(config_path, config)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = ZipImageDataset(zip_path=args.data_zip, subset='test', transform=test_transform)
    
    # Verify datasets are not empty
    if len(test_dataset) == 0:
        raise ValueError(f"No images found in {args.data_zip} for val subset")
    
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"Test dataset classes: {class_names} (total: {num_classes})")
    print(f"Test samples: {len(test_dataset)}")
    
    for model_config in config['models']:
        model_name = model_config['name']
        model_class = model_config['class']
        test_model(model_name, model_class, config, device, test_loader, num_classes)

if __name__ == "__main__":
    main()