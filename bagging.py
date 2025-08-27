import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import gc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import os
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import time
import traceback
import json
import seaborn as sns
import argparse
import logging

from FastKAN_model import ConvNeXtFastKAN, BasicCNNFastKAN, DenseNetFastKAN, ResNetFastKAN, EfficientNetV2FastKAN, MobileNetV2FastKAN
from KAN_model import ConvNeXtKAN, BasicCNNKAN, DenseNetKAN, ResNetKAN, EfficientNetV2KAN, MobileNetV2KAN

from utils import CheckpointManager, ZipImageDataset, EarlyStopping

def _safe_train_single_model(model, train_loader, val_loader, epochs, learning_rate, model_num, device, 
                            checkpoint_dir, save_models=True, log_file=None):
    """
    Safely train a single model with checkpointing, early stopping, and error handling.
    
    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (optional).
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        model_num: Model identifier for logging and saving.
        device: Device to train on (e.g., 'cuda' or 'cpu').
        checkpoint_dir: Directory to save checkpoints.
        save_models: Whether to save model checkpoints and final state.
        log_file: Path to log file for CheckpointManager (optional).
    
    Returns:
        dict: Training history with train_loss, train_acc, val_loss, val_acc, or None if training fails.
    """
    try:
        # Initialize criterion, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Initialize CheckpointManager
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model_name=f"model_{model_num}",
            keep_last_n=5,
            log_file=log_file
        )
        
        # Initialize EarlyStopping
        early_stopping = EarlyStopping(patience=7, min_delta=0.001, restore_best_weights=True)
        
        # Initialize history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        total_training_time = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            try:
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                num_train_batches = len(train_loader)
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    try:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                        
                        optimizer.zero_grad(set_to_none=True)  # Optimize memory usage
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        train_total += target.size(0)
                        train_correct += (predicted == target).sum().item()
                        
                        del data, target, output, loss, predicted
                        
                        batch_progress = (batch_idx + 1) / num_train_batches * 100
                        print(f"\r[Model {model_num}] Epoch {epoch+1}/{epochs} Batch {batch_idx+1}/{len(train_loader)} Train Loss: {train_loss/(batch_idx+1):.4f} Train Acc: {train_correct/train_total:.4f} Progress: [{batch_progress:.1f}%]", end='', flush=True)
                        
                        if batch_idx % 50 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"[Model {model_num}] OOM at batch {batch_idx}, clearing cache...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise e
                
                train_accuracy = train_correct / train_total if train_total > 0 else 0
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
                print(f'\n[Model {model_num}] Epoch {epoch+1}/{epochs} completed. Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
                
                print(f'Validating...')
                # Validation phase
                val_accuracy = 0.0
                avg_val_loss = 0.0
                num_val_batches = len(val_loader) if val_loader else 0
                if val_loader:
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(val_loader):
                            try:
                                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                                output = model(data)
                                loss = criterion(output, target)
                                
                                val_loss += loss.item()
                                _, predicted = torch.max(output.data, 1)
                                val_total += target.size(0)
                                val_correct += (predicted == target).sum().item()
                                
                                del data, target, output, loss, predicted
                                
                                batch_progress = (batch_idx + 1) / num_val_batches * 100
                                print(f'\r[Model {model_num}] Validation Batch {batch_idx+1}/{len(val_loader)} Val Loss: {val_loss/(batch_idx+1):.4f} Val Acc: {val_correct/val_total:.4f} Progress: [{batch_progress:.1f}%]',  end='', flush=True)
                                
                                if batch_idx % 50 == 0 and torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    print(f"[Model {model_num}] Validation OOM, skipping batch...")
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue
                                else:
                                    raise e
                    
                    val_accuracy = val_correct / val_total if val_total > 0 else 0
                    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
                    
                    print(f'[Model {model_num}] Validation Loss: {avg_val_loss:.4f}, ' f'Validation Accuracy: {val_accuracy:.4f}')
                
                scheduler.step()
                
                # Update history
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_accuracy)
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_accuracy)
                
                total_training_time = time.time() - start_time
                
                # Save checkpoint every 10 epochs or at the last epoch
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss
                if save_models and ((epoch + 1) % 10 == 0 or (epoch + 1) == epochs):
                    checkpoint_manager.save_checkpoint(
                        epoch=epoch + 1,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_losses=history['train_loss'],
                        val_losses=history['val_loss'],
                        train_accs=history['train_acc'],
                        val_accs=history['val_acc'],
                        best_val_loss=best_val_loss,
                        total_time=total_training_time,
                        is_best=is_best
                    )
                
                # Early stopping check
                if val_loader and early_stopping(avg_val_loss, model):
                    print(f"[Model {model_num}] Early stopping triggered at epoch {epoch+1}")
                    break
                
                print(f'[Model {model_num}] Epoch {epoch+1}/{epochs}: Final Train Loss: {avg_train_loss:.4f}, Final Train Acc: {train_accuracy:.4f} Final Val Loss: {avg_val_loss:.4f}, Final Val Acc: {val_accuracy: 4f}')
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"[Model {model_num}] Error in epoch {epoch+1}: {str(e)}")
                logging.error(f"[Model {model_num}] Epoch {epoch+1} failed: {str(e)}")
                continue
        
        # Save final model state
        final_model_path = None
        if save_models:
            final_model_path = os.path.join(checkpoint_dir, f"model_{model_num}_final.pth")
            torch.save(model.state_dict(), final_model_path)
            print(f"[Model {model_num}] Final model saved to {final_model_path}")
        
        # Save training history
        history_file = os.path.join(checkpoint_dir, f"model_{model_num}_history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[Model {model_num}] Training history saved to {history_file}")
        
        return history
    
    except Exception as e:
        error_msg = f"[Model {model_num}] Training failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        logging.error(error_msg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None

def safe_train_model_worker(args):
    try:
        (model_id, bootstrap_indices, model_kwargs, epochs, batch_size, 
        learning_rate, save_models, result_dir, train_dataset_info, 
        val_dataset_info, random_seed, num_workers, gpu_id) = args
        
        print(f"[Model {model_id}] Starting training process...")
        
        torch.manual_seed(random_seed + model_id)
        np.random.seed(random_seed + model_id)
        random.seed(random_seed + model_id)
        
        if torch.cuda.is_available() and gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            device = torch.device("cpu")
        
        print(f"[Model {model_id}] Using device: {device}")
        
        # Initialize datasets
        train_dataset = ZipImageDataset(
            zip_path=train_dataset_info['zip_path'],
            subset='train',
            transform=train_dataset_info['transform']
        )
        
        val_dataset = None
        if val_dataset_info:
            val_dataset = ZipImageDataset(
                zip_path=val_dataset_info['zip_path'],
                subset='val',
                transform=val_dataset_info['transform']
            )
        
        bootstrap_dataset = Subset(train_dataset, bootstrap_indices)
        print(f"[Model {model_id}] Bootstrap sample size: {len(bootstrap_dataset)}")
        
        train_loader = DataLoader(
            bootstrap_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size,
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=False
            )
        
        print(f"[Model {model_id}] Initializing model...")
        model = ConvNeXtFastKAN(**model_kwargs)
        model = model.to(device)
        
        print(f"[Model {model_id}] Starting training...")
        checkpoint_dir = os.path.join(result_dir, f"model_{model_id}_checkpoints")
        log_file = os.path.join(result_dir, f"model_{model_id}_checkpoint.log")
        start_time = time.time()
        history = _safe_train_single_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            model_num=model_id,
            device=device,
            checkpoint_dir=checkpoint_dir,
            save_models=save_models,
            log_file=log_file
        )
        
        final_val_acc = 0.0
        if history and 'val_acc' in history and history['val_acc']:
            final_val_acc = max(history['val_acc'])
        
        # Save model performance metrics
        if history and save_models:
            performance_file = os.path.join(checkpoint_dir, f"model_{model_id}_performance.json")
            performance_metrics = {
                'model_id': model_id,
                'final_val_acc': final_val_acc,
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'final_train_acc': history['train_acc'][-1] if history['train_acc'] else None,
                'training_time': time.time() - start_time
            }
            with open(performance_file, 'w') as f:
                json.dump(performance_metrics, f, indent=2)
            print(f"[Model {model_id}] Performance metrics saved to {performance_file}")
        
        del model
        del train_loader
        if val_loader:
            del val_loader
        del train_dataset
        if val_dataset:
            del val_dataset
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        
        gc.collect()
        
        print(f"[Model {model_id}] Training completed successfully! Final val acc: {final_val_acc:.4f}")
        
        return {
            'model_id': model_id,
            'history': history,
            'model_path': os.path.join(checkpoint_dir, f"model_{model_id}_final.pth") if save_models else None,
            'final_val_acc': final_val_acc,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        error_msg = f"Model {model_id} failed: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        logging.error(error_msg)
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        
        return {
            'model_id': model_id,
            'history': None,
            'model_path': None,
            'final_val_acc': 0.0,
            'success': False,
            'error': error_msg
        }

def evaluate_single_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    f1_micro = f1_score(all_targets, all_predictions, average='micro')
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
    recall_macro = recall_score(all_targets, all_predictions, average='macro')
    recall_micro = recall_score(all_targets, all_predictions, average='micro')
    recall_weighted = recall_score(all_targets, all_predictions, average='weighted')
    precision_macro = precision_score(all_targets, all_predictions, average='macro')
    precision_micro = precision_score(all_targets, all_predictions, average='micro')
    precision_weighted = precision_score(all_targets, all_predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'recall_weighted': recall_weighted,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'precision_weighted': precision_weighted,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    return metrics

class BaggingEnsemble:
    def __init__(self, base_model_class, model_kwargs, n_estimators=5, subsample_ratio=0.8, 
                bootstrap=True, random_state=42, max_workers=None, use_threading=False, checkpoint_dir=None):
        self.base_model_class = base_model_class
        self.model_kwargs = model_kwargs
        self.n_estimators = n_estimators
        self.subsample_ratio = subsample_ratio
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.checkpoint_dir = checkpoint_dir  # Use checkpoint_dir consistently
        self.use_threading = use_threading
        
        if max_workers is None:
            if use_threading:
                self.max_workers = min(n_estimators, 2)
            else:
                self.max_workers = min(n_estimators, max(1, mp.cpu_count() // 2))
        else:
            self.max_workers = max_workers
        
        self.models = []
        self.model_paths = []
        self.model_performances = []
        self.best_model_id = None
        self.best_model_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)
        
        print(f"Initialized Bagging Ensemble:")
        print(f"  - Number of estimators: {n_estimators}")
        print(f"  - Max workers: {self.max_workers}")
        print(f"  - Using threading: {use_threading}")
        print(f"  - Subsample ratio: {subsample_ratio}")
        print(f"  - Bootstrap sampling: {bootstrap}")
        print(f"  - Checkpoint directory: {checkpoint_dir or 'Not specified, using model-specific paths'}")
        
    def _create_bootstrap_indices(self, dataset_size):
        sample_size = int(dataset_size * self.subsample_ratio)
        
        if self.bootstrap:
            indices = np.random.choice(dataset_size, size=sample_size, replace=True)
        else:
            indices = np.random.choice(dataset_size, size=sample_size, replace=False)
            
        return indices.tolist()
    
    def _get_dataset_info(self, dataset):
        return {
            'zip_path': dataset.zip_path,
            'transform': dataset.transform
        }
    
    def fit(self, train_dataset, val_dataset=None, epochs=10, batch_size=32, 
            learning_rate=0.001, save_models=True, result_dir="bagging_models"):
        if save_models:
            os.makedirs(result_dir, exist_ok=True)
        
        dataset_size = len(train_dataset)
        training_history = [None] * self.n_estimators
        
        print(f"\nTraining {self.n_estimators} models...")
        print(f"Original dataset size: {dataset_size}")
        print(f"Using {'threading' if self.use_threading else 'multiprocessing'}")
        print(f"Max workers: {self.max_workers}")
        print(f"{'='*60}")
        
        train_dataset_info = self._get_dataset_info(train_dataset)
        val_dataset_info = self._get_dataset_info(val_dataset) if val_dataset else None
        
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Available GPUs: {num_gpus}")
        
        model_args = []
        for i in range(self.n_estimators):
            bootstrap_indices = self._create_bootstrap_indices(dataset_size)
            gpu_id = i % num_gpus if num_gpus > 0 else None
            
            # Use self.checkpoint_dir if provided, else fall back to result_dir
            checkpoint_dir = self.checkpoint_dir if self.checkpoint_dir else os.path.join(result_dir, f"model_{i+1}_checkpoints")
            
            args = (
                i + 1,
                bootstrap_indices,
                self.model_kwargs,
                epochs,
                batch_size,
                learning_rate,
                save_models,
                result_dir,
                train_dataset_info,
                val_dataset_info,
                self.random_state,
                gpu_id
            )
            model_args.append(args)
        
        start_time = time.time()
        completed_models = 0
        failed_models = 0
        model_results = []
        
        executor_class = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            future_to_model = {
                executor.submit(safe_train_model_worker, args): args[0] 
                for args in model_args
            }
            
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                
                try:
                    result = future.result(timeout=7200)
                    
                    if result['success']:
                        training_history[model_id - 1] = result['history']
                        model_results.append(result)
                        if result['model_path']:
                            self.model_paths.append(result['model_path'])
                        
                        completed_models += 1
                        elapsed_time = time.time() - start_time
                        print(f"\n✓ Model {model_id} completed! "
                            f"Val Acc: {result['final_val_acc']:.4f} "
                            f"({completed_models}/{self.n_estimators}) "
                            f"[{elapsed_time:.1f}s elapsed]")
                    else:
                        failed_models += 1
                        print(f"\n✗ Model {model_id} failed: {result['error']}")
                        logging.error(f"Model {model_id} failed: {result['error']}")
                        
                except Exception as e:
                    failed_models += 1
                    print(f"\n✗ Model {model_id} failed with exception: {str(e)}")
                    logging.error(f"Model {model_id} failed with exception: {str(e)}")
        
        if model_results:
            best_result = max(model_results, key=lambda x: x['final_val_acc'])
            self.best_model_id = best_result['model_id']
            self.best_model_path = best_result['model_path']
            
            print(f"\n🏆 Best Model: Model {self.best_model_id} with validation accuracy: {best_result['final_val_acc']:.4f}")
            
            self.model_performances = [
                {
                    'model_id': result['model_id'],
                    'final_val_acc': result['final_val_acc'],
                    'model_path': result['model_path'],
                    'final_train_loss': result['history']['train_loss'][-1] if result['history'] and result['history']['train_loss'] else None,
                    'final_val_loss': result['history']['val_loss'][-1] if result['history'] and result['history']['val_loss'] else None,
                    'final_train_acc': result['history']['train_acc'][-1] if result['history'] and result['history']['train_acc'] else None
                }
                for result in model_results
            ]
            
            self.model_performances.sort(key=lambda x: x['final_val_acc'], reverse=True)
            
            performance_file = os.path.join(result_dir, "model_performance_summary.json")
            with open(performance_file, 'w') as f:
                json.dump({
                    'best_model_id': self.best_model_id,
                    'best_model_path': self.best_model_path,
                    'model_performances': self.model_performances
                }, f, indent=2)
            
            print(f"Model performance summary saved to: {performance_file}")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Successfully trained: {completed_models}/{self.n_estimators} models")
        print(f"Failed models: {failed_models}")
        print(f"{'='*60}")
        
        if completed_models == 0:
            raise RuntimeError("All models failed to train!")
        
        self._load_trained_models()
        
        return [h for h in training_history if h is not None]
    
    def _load_trained_models(self):
        self.models = []
        
        for model_path in self.model_paths:
            if os.path.exists(model_path):
                try:
                    model = self.base_model_class(**self.model_kwargs)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    self.models.append(model)
                    print(f"✓ Loaded model from {model_path}")
                except Exception as e:
                    print(f"✗ Failed to load model from {model_path}: {str(e)}")
                    logging.error(f"Failed to load model from {model_path}: {str(e)}")
            else:
                print(f"✗ Model file not found: {model_path}")
                logging.error(f"Model file not found: {model_path}")
    
    def get_best_model(self):
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            raise ValueError("Best model not found or not saved")
        
        best_model = self.base_model_class(**self.model_kwargs)
        best_model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        best_model.to(self.device)
        return best_model
    
    def evaluate_individual_models(self, test_loader, class_names=None):
        if not self.models:
            raise ValueError("No trained models found. Please train the ensemble first.")
        
        print(f"\n{'='*60}")
        print("EVALUATING INDIVIDUAL MODELS")
        print(f"{'='*60}")
        
        individual_results = []
        
        for i, model in enumerate(self.models):
            print(f"\nEvaluating Model {i+1}...")
            metrics = evaluate_single_model(model, test_loader, self.device)
            
            result = {
                'model_id': i + 1,
                'metrics': metrics
            }
            individual_results.append(result)
            
            print(f"Model {i+1} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
            print(f"  F1 (Micro): {metrics['f1_micro']:.4f}")
            print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
            print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
            print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
            
            if class_names:
                print(f"Classification Report for Model {i+1}:")
                print(classification_report(metrics['targets'], metrics['predictions'], target_names=class_names))
        
        best_test_result = max(individual_results, key=lambda x: x['metrics']['f1_macro'])
        
        print(f"\n🏆 BEST MODEL ON TEST SET:")
        print(f"Model {best_test_result['model_id']} with F1 macro: {best_test_result['metrics']['f1_macro']:.4f}")
        
        return individual_results
    
    def predict(self, test_loader, voting='soft'):
        if not self.models:
            raise ValueError("No trained models found. Please train the ensemble first.")
        
        print(f"Making predictions with {len(self.models)} models...")
        
        all_predictions = []
        all_probabilities = []
        
        for i, model in enumerate(self.models):
            model.eval()
            model_predictions = []
            model_probabilities = []
            
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(self.device)
                    outputs = model(data)
                    
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    model_predictions.extend(predicted.cpu().numpy())
                    model_probabilities.extend(probabilities.cpu().numpy())
            
            all_predictions.append(model_predictions)
            all_probabilities.append(model_probabilities)
            print(f"✓ Model {i+1} predictions completed")
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        if voting == 'soft':
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
            return ensemble_predictions, all_predictions, ensemble_probabilities
        else:
            ensemble_predictions = []
            for i in range(all_predictions.shape[1]):
                votes = all_predictions[:, i]
                ensemble_predictions.append(np.bincount(votes).argmax())
            
            return np.array(ensemble_predictions), all_predictions, None
    
    def evaluate(self, test_loader, true_labels=None, class_names=None, save_dir="bagging_training_result"):
        if not self.models:
            raise ValueError("No trained models found. Please train the ensemble first.")
        
        print(f"\n{'='*60}")
        print("EVALUATING ENSEMBLE")
        print(f"{'='*60}")
        
        individual_results = self.evaluate_individual_models(test_loader, class_names)
        
        predictions, individual_preds, probabilities = self.predict(test_loader, voting='soft')
        
        if true_labels is None:
            true_labels = []
            for _, labels in test_loader:
                true_labels.extend(labels.numpy())
            true_labels = np.array(true_labels)
        
        ensemble_accuracy = accuracy_score(true_labels, predictions)
        ensemble_f1_macro = f1_score(true_labels, predictions, average='macro')
        ensemble_f1_micro = f1_score(true_labels, predictions, average='micro')
        ensemble_f1_weighted = f1_score(true_labels, predictions, average='weighted')
        ensemble_recall_macro = recall_score(true_labels, predictions, average='macro')
        ensemble_recall_micro = recall_score(true_labels, predictions, average='micro')
        ensemble_recall_weighted = recall_score(true_labels, predictions, average='weighted')
        ensemble_precision_macro = precision_score(true_labels, predictions, average='macro')
        ensemble_precision_micro = precision_score(true_labels, predictions, average='micro')
        ensemble_precision_weighted = precision_score(true_labels, predictions, average='weighted')
        
        individual_metrics = {
            'accuracy': np.mean([r['metrics']['accuracy'] for r in individual_results]),
            'f1_macro': np.mean([r['metrics']['f1_macro'] for r in individual_results]),
            'f1_micro': np.mean([r['metrics']['f1_micro'] for r in individual_results]),
            'f1_weighted': np.mean([r['metrics']['f1_weighted'] for r in individual_results]),
            'recall_macro': np.mean([r['metrics']['recall_macro'] for r in individual_results]),
            'recall_micro': np.mean([r['metrics']['recall_micro'] for r in individual_results]),
            'recall_weighted': np.mean([r['metrics']['recall_weighted'] for r in individual_results]),
            'precision_macro': np.mean([r['metrics']['precision_macro'] for r in individual_results]),
            'precision_micro': np.mean([r['metrics']['precision_micro'] for r in individual_results]),
            'precision_weighted': np.mean([r['metrics']['precision_weighted'] for r in individual_results])
        }
        
        print(f"\nEnsemble Evaluation Results:")
        print(f"{'='*50}")
        print(f"Ensemble Metrics:")
        print(f"  Accuracy: {ensemble_accuracy:.4f}")
        print(f"  F1 (Macro): {ensemble_f1_macro:.4f}")
        print(f"  F1 (Micro): {ensemble_f1_micro:.4f}")
        print(f"  F1 (Weighted): {ensemble_f1_weighted:.4f}")
        print(f"  Recall (Macro): {ensemble_recall_macro:.4f}")
        print(f"  Recall (Micro): {ensemble_recall_micro:.4f}")
        print(f"  Recall (Weighted): {ensemble_recall_weighted:.4f}")
        print(f"  Precision (Macro): {ensemble_precision_macro:.4f}")
        print(f"  Precision (Micro): {ensemble_precision_micro:.4f}")
        print(f"  Precision (Weighted): {ensemble_precision_weighted:.4f}")
        
        print(f"\nAverage Individual Model Metrics:")
        print(f"{'='*50}")
        print(f"  Accuracy: {individual_metrics['accuracy']:.4f}")
        print(f"  F1 (Macro): {individual_metrics['f1_macro']:.4f}")
        print(f"  F1 (Micro): {individual_metrics['f1_micro']:.4f}")
        print(f"  F1 (Weighted): {individual_metrics['f1_weighted']:.4f}")
        print(f"  Recall (Macro): {individual_metrics['recall_macro']:.4f}")
        print(f"  Recall (Micro): {individual_metrics['recall_micro']:.4f}")
        print(f"  Recall (Weighted): {individual_metrics['recall_weighted']:.4f}")
        print(f"  Precision (Macro): {individual_metrics['precision_macro']:.4f}")
        print(f"  Precision (Micro): {individual_metrics['precision_micro']:.4f}")
        print(f"  Precision (Weighted): {individual_metrics['precision_weighted']:.4f}")
        
        if class_names:
            print(f"\nClassification Report for Ensemble:")
            print(classification_report(true_labels, predictions, target_names=class_names))
        
        # Save confusion matrix plot
        if class_names:
            print("\nGenerating and saving Confusion Matrix for Ensemble...")
            cm = confusion_matrix(true_labels, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix for Ensemble')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_save_path = os.path.join(save_dir, 'ensemble_confusion_matrix.png')
            plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to: {cm_save_path}")
        
        results = {
            'ensemble_metrics': {
                'accuracy': ensemble_accuracy,
                'f1_macro': ensemble_f1_macro,
                'f1_micro': ensemble_f1_micro,
                'f1_weighted': ensemble_f1_weighted,
                'recall_macro': ensemble_recall_macro,
                'recall_micro': ensemble_recall_micro,
                'recall_weighted': ensemble_recall_weighted,
                'precision_macro': ensemble_precision_macro,
                'precision_micro': ensemble_precision_micro,
                'precision_weighted': ensemble_precision_weighted
            },
            'individual_metrics': individual_metrics,
            'individual_results': individual_results,
            'ensemble_predictions': predictions.tolist(),
            'individual_predictions': individual_preds.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'confusion_matrix': cm.tolist() if class_names else None,
            'class_names': class_names
        }
        
        return results
    
    def save_ensemble(self, save_dir="bagging_ensemble"):
        os.makedirs(save_dir, exist_ok=True)
        
        ensemble_state = {
            'n_estimators': self.n_estimators,
            'subsample_ratio': self.subsample_ratio,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'model_kwargs': self.model_kwargs,
            'model_paths': self.model_paths,
            'best_model_id': self.best_model_id,
            'best_model_path': self.best_model_path,
            'model_performances': self.model_performances
        }
        
        state_file = os.path.join(save_dir, "ensemble_state.json")
        with open(state_file, 'w') as f:
            json.dump(ensemble_state, f, indent=2)
        
        print(f"Ensemble state saved to: {state_file}")

    def load_ensemble(self, save_dir="bagging_ensemble"):
        state_file = os.path.join(save_dir, "ensemble_state.json")
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"Ensemble state file not found: {state_file}")
        
        with open(state_file, 'r') as f:
            ensemble_state = json.load(f)
        
        self.n_estimators = ensemble_state['n_estimators']
        self.subsample_ratio = ensemble_state['subsample_ratio']
        self.bootstrap = ensemble_state['bootstrap']
        self.random_state = ensemble_state['random_state']
        self.model_kwargs = ensemble_state['model_kwargs']
        self.model_paths = ensemble_state['model_paths']
        self.best_model_id = ensemble_state['best_model_id']
        self.best_model_path = ensemble_state['best_model_path']
        self.model_performances = ensemble_state['model_performances']
        
        self._load_trained_models()
        
        print(f"Ensemble loaded from: {state_file}")

    def plot_training_history(self, history_list, save_dir="training_history_plots", dpi=300, image_format="png"):
        os.makedirs(save_dir, exist_ok=True)
        
        for i, history in enumerate(history_list):
            if history is None:
                print(f"Skipping Model {i+1}: No history available")
                continue
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_acc'], label='train', color='blue', linewidth=2)
            if history['val_acc']:
                plt.plot(history['val_acc'], '--', label='valid', color='orange', linewidth=2)
            plt.title(f'Model {i+1} Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.xticks(range(0, len(history['train_acc']), max(1, len(history['train_acc'])//5)))
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_loss'], label='train', color='blue', linewidth=2)
            if history['val_loss']:
                plt.plot(history['val_loss'], '--', label='valid', color='orange', linewidth=2)
            plt.title(f'Model {i+1} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim(0, max(max(history['train_loss'], default=2), max(history['val_loss'], default=2)) * 1.1)
            plt.xticks(range(0, len(history['train_loss']), max(1, len(history['train_loss'])//5)))
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"model_{i+1}_history.{image_format}")
            plt.savefig(save_path, format=image_format, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Training history plot for Model {i+1} saved to: {save_path}")

    def plot_model_performance(self, save_dir="model_performance_plots", dpi=300, image_format="png"):
        os.makedirs(save_dir, exist_ok=True)
        
        model_ids = [p['model_id'] for p in self.model_performances]
        val_accs = [p['final_val_acc'] for p in self.model_performances]
        
        plt.figure(figsize=(10, 6))
        plt.bar(model_ids, val_accs, color='skyblue')
        plt.xlabel('Model ID')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy of Individual Models')
        plt.xticks(model_ids)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path = os.path.join(save_dir, f"model_performance.{image_format}")
        plt.savefig(save_path, format=image_format, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Model performance plot saved to: {save_path}")

    def cleanup(self):
        if self.models:
            for model in self.models:
                del model
            self.models = []
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        print("Cleaned up models and freed memory.")

def get_model_registry():
    """Return a dictionary mapping model names to their classes."""
    return {
        'convnext_fastkan': ConvNeXtFastKAN,
        'basiccnn_fastkan': BasicCNNFastKAN,
        'densenet_fastkan': DenseNetFastKAN,
        'resnet_fastkan': ResNetFastKAN,
        'efficientnetv2_fastkan': EfficientNetV2FastKAN,
        'mobilenetv2_fastkan': MobileNetV2FastKAN,
        'convnext_kan': ConvNeXtKAN,
        'basic_cnn_kan': BasicCNNKAN,
        'densenet_kan': DenseNetKAN,
        'resnet_kan': ResNetKAN,
        'efficientnetv2_kan': EfficientNetV2KAN,
        'mobilenetv2_kan': MobileNetV2KAN
    }

def save_training_config(config_path, config):
    """Save config to JSON, excluding non-serializable objects."""
    serializable_config = config.copy()
    
    # Convert 'base_model_class' to string name if present
    if 'base_model_class' in serializable_config:
        serializable_config['base_model_class'] = serializable_config['base_model_class'].__name__
    
    # Convert 'class' fields in 'models' list to string names if present
    if 'models' in serializable_config:
        serializable_config['models'] = [
            {
                'name': model['name'],
                'class': model['class'].__name__
            } for model in serializable_config['models']
        ]
    
    try:
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
    except (IOError, PermissionError) as e:
        logging.error(f"Failed to save config to {config_path}: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="Train a bagging ensemble with multiple FastKAN models using a zip file dataset.")
    parser.add_argument('--data-dir', type=str, default="data.zip", help='Path to the zip file containing the dataset.')
    parser.add_argument('--model', type=str, help='Specific model to train (optional)')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the training configuration file.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])
    
    default_config = {
        'n_estimators': 10,
        'epochs': 75,
        'batch_size': 128,
        'learning_rate': 0.001,
        'subsample_ratio': 0.8,
        'max_workers': 4,
        'use_threading': False,
        'result_dir': 'bagging_training_results',
        'plot_save_dir': 'training_history_plots',
        'data_dir': 'data.zip',
        'checkpoint_dir': 'bagging_checkpoints',  # Updated to match saved config directory
        'image_format': 'png',
        'dpi': 300,
        'no_save_models': False,
        'random_state': 42,
        'model_kwargs': {
            'hidden_dims': [512, 256],
            'num_classes': 11,
            'pretrained': True,
            'freeze_backbone': True
        },
        'models': [
            {'name': 'convnext_fastkan', 'class': ConvNeXtFastKAN},
            {'name': 'basic_cnn_fastkan', 'class': BasicCNNFastKAN},
            {'name': 'densenet_fastkan', 'class': DenseNetFastKAN},
            {'name': 'resnet_fastkan', 'class': ResNetFastKAN},
            {'name': 'efficientnetv2_fastkan', 'class': EfficientNetV2FastKAN},
            {'name': 'mobilenetv2_fastkan', 'class': MobileNetV2FastKAN},
            {'name': 'convnext_kan', 'class': ConvNeXtKAN},
            {'name': 'basic_cnn_kan', 'class': BasicCNNKAN},
            {'name': 'densenet_kan', 'class': DenseNetKAN}, 
            {'name': 'resnet_kan', 'class': ResNetKAN},
            {'name': 'efficientnetv2_kan', 'class': EfficientNetV2KAN},
            {'name': 'mobilenetv2_kan', 'class': MobileNetV2KAN}
        ]
    }
    
    os.makedirs(default_config['result_dir'], exist_ok=True)
    os.makedirs(default_config['plot_save_dir'], exist_ok=True) 
    os.makedirs(default_config['checkpoint_dir'], exist_ok=True)
    
    model_registry = get_model_registry()
    print("Available models:")
    for model_name, model_class in model_registry.items():
        print(f"  - {model_name}: {model_class.__name__}")
    
    # Load global config if provided
    config = default_config.copy()
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                if 'model_kwargs' in loaded_config:
                    config['model_kwargs'].update(loaded_config['model_kwargs'])
                logging.info(f"Loaded global config from {args.config}")
                print(f"Loaded global config from {args.config}")
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing global config file {args.config}: {e}, using default config")
            print(f"Error parsing global config file {args.config}: {e}, using default config")
        except PermissionError as e:
            logging.error(f"Permission denied accessing global config file {args.config}: {e}, using default config")
            print(f"Permission denied accessing global config file {args.config}: {e}, using default config")
    else:
        logging.info(f"No global config file provided or found at {args.config}, using default config")
        print(f"No global config file provided or found at {args.config}, using default config")
    
    # Override data_dir if provided
    if args.data_dir and os.path.exists(args.data_dir):
        config['data_dir'] = args.data_dir
    
    # Filter models if specific model is provided
    if args.model:
        config['models'] = [m for m in config['models'] if m['name'] == args.model]
        if not config['models']:
            raise ValueError(f"Model {args.model} not found in configuration")
    
    if not os.path.exists(config['data_dir']):
        raise FileNotFoundError(f"Data zip file not found at {config['data_dir']}")
    
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
    
    train_dataset = ZipImageDataset(zip_path=config['data_dir'], subset='train', transform=train_transform)
    val_dataset = ZipImageDataset(zip_path=config['data_dir'], subset='val', transform=val_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"No images found in {config['data_dir']} for {'train' if len(train_dataset) == 0 else 'val'} subset")
    
    num_classes = len(train_dataset.classes)
    config['model_kwargs']['num_classes'] = num_classes
    
    class_names = train_dataset.classes 
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    for model_config in config['models']:
        model_name = model_config['name']
        model_class = model_config['class']
        print(f"\n=== Training {model_name} Ensemble ===")
        
        # Use consistent directory for model-specific config
        model_config_path = os.path.join(config['result_dir'], f"{model_name}_config.json")
        model_specific_config = config.copy()
        if os.path.exists(model_config_path):
            try:
                with open(model_config_path, 'r') as f:
                    loaded_config = json.load(f)
                    model_specific_config.update(loaded_config)
                    if 'model_kwargs' in loaded_config:
                        model_specific_config['model_kwargs'].update(loaded_config['model_kwargs'])
                    model_specific_config['base_model_class'] = model_class
                    logging.info(f"Loaded model-specific config for {model_name} from {model_config_path}")
                    print(f"Loaded model-specific config for {model_name} from {model_config_path}")
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing config file {model_config_path}: {e}, using default config")
                print(f"Error parsing config file {model_config_path}: {e}, using default config")
            except PermissionError as e:
                logging.error(f"Permission denied accessing config file {model_config_path}: {e}, using default config")
                print(f"Permission denied accessing config file {model_config_path}: {e}, using default config")
        else:
            logging.warning(f"No config file found for {model_name} at {model_config_path}, using default config")
            print(f"No config file found for {model_name} at {model_config_path}, using default config")
            model_specific_config['base_model_class'] = model_class
        
        model_specific_config['model_kwargs']['num_classes'] = num_classes
        
        save_config_path = os.path.join(config['result_dir'], f"{model_name}_training_config.json")
        save_training_config(save_config_path, model_specific_config)
        logging.info(f"Saved {model_name} config to {save_config_path}")
        print(f"Saved {model_name} config to {save_config_path}")
        
        try:
            ensemble = BaggingEnsemble(
                base_model_class=model_specific_config['base_model_class'],
                model_kwargs=model_specific_config['model_kwargs'],
                n_estimators=model_specific_config['n_estimators'],
                subsample_ratio=model_specific_config['subsample_ratio'],
                bootstrap=True,
                checkpoint_dir=model_specific_config['checkpoint_dir'],  # Use checkpoint_dir from config
                random_state=model_specific_config['random_state'],
                max_workers=model_specific_config['max_workers'],
                use_threading=model_specific_config['use_threading']
            )
            
            training_history = ensemble.fit(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=model_specific_config['epochs'],
                batch_size=model_specific_config['batch_size'],
                learning_rate=model_specific_config['learning_rate'],
                save_models=not model_specific_config['no_save_models'],
                result_dir=model_specific_config['result_dir']
            )
            
            ensemble.save_ensemble(save_dir=os.path.join(model_specific_config['result_dir'], model_name))
            
            if training_history:
                ensemble.plot_training_history(
                    training_history,
                    save_dir=os.path.join(model_specific_config['plot_save_dir'], model_name),
                    dpi=model_specific_config['dpi'],
                    image_format=model_specific_config['image_format']
                )
                ensemble.plot_model_performance(
                    save_dir=os.path.join(model_specific_config['plot_save_dir'], model_name),
                    dpi=model_specific_config['dpi'],
                    image_format=model_specific_config['image_format']
                )
                
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            test_dataset = ZipImageDataset(zip_path=config['data_dir'], subset='test', transform=test_transform)
            
            if len(test_dataset) == 0:
                raise ValueError(f"No images found in {config['data_dir']} for test subset")
            
            test_loader = DataLoader(test_dataset, batch_size=model_specific_config['batch_size'], shuffle=False, num_workers=4)
            
            results = ensemble.evaluate(
                test_loader,
                class_names=class_names,
                save_dir=os.path.join(config['result_dir'], model_name)
            )
            print(f"\n{model_name} Ensemble Evaluation Results:")
            print(f"Accuracy: {results['ensemble_metrics']['accuracy']:.4f}")
            print(f"F1 (Macro): {results['ensemble_metrics']['f1_macro']:.4f}")
            print(f"Best individual model F1 (Macro): {max([r['metrics']['f1_macro'] for r in results['individual_results']]):.4f}")
            
            print(f"Confusion Matrix saved to: {os.path.join(config['result_dir'], model_name, 'ensemble_confusion_matrix.png')}")
            print(f"Individual model results saved to: {os.path.join(config['result_dir'], model_name, 'individual_results.json')}")
            print(f"Ensemble predictions saved to: {os.path.join(config['result_dir'], model_name, 'ensemble_predictions.json')}")
            print(f"Individual predictions saved to: {os.path.join(config['result_dir'], model_name, 'individual_predictions.json')}")
            print(f"Probabilities saved to: {os.path.join(config['result_dir'], model_name, 'probabilities.json')}")
            print(f"Class names: {results['class_names']}")
            print(f"Ensemble metrics: {results['ensemble_metrics']}")
            
            # Save results to JSON
            results_path = os.path.join(config['result_dir'], f"{model_name}_evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Saved {model_name} evaluation results to {results_path}")
            print(f"Saved {model_name} evaluation results to {results_path}")
            
            ensemble.cleanup()
            
        except Exception as e:
            logging.error(f"Training failed for {model_name}: {str(e)}")
            print(f"Training failed for {model_name}: {str(e)}")
            continue
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\nAll model training and evaluation completed!")

if __name__ == "__main__":
    main()