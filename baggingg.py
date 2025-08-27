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

from FastKAN_model import EfficientNetV2FastKAN
from utils import ZipImageDataset, EarlyStopping, CheckpointManager

def _safe_train_single_model(model, train_loader, val_loader, epochs, learning_rate, model_num, device, 
                            checkpoint_dir, save_models=True, log_file=None):
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            logging.info(f"[Model {model_num}] Created checkpoint directory: {checkpoint_dir}")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"[Model {model_num}] Created checkpoint directory: {checkpoint_dir}\n")
        except PermissionError as e:
            logging.error(f"[Model {model_num}] Failed to create checkpoint directory {checkpoint_dir}: {e}")
            raise
        
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model_name=f"model_{model_num}",
            keep_last_n=3,  # Reduced to keep last 3 checkpoints
            log_file=log_file
        )
        
        early_stopping = EarlyStopping(patience=7, min_delta=0.001, restore_best_weights=True)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        total_training_time = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            try:
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                num_train_batches = len(train_loader)
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    try:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                        
                        optimizer.zero_grad(set_to_none=True)
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
                    
                    print(f'[Model {model_num}] Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
                
                scheduler.step()
                
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_accuracy)
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_accuracy)
                
                total_training_time = time.time() - start_time
                
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss
                
                if save_models:
                    try:
                        checkpoint_path = os.path.join(checkpoint_dir, f"model_{model_num}_epoch_{epoch+1}.pth")
                        print(f"[Model {model_num}] Saving checkpoint to {checkpoint_path}")
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
                            is_best=is_best,
                            
                        )
                        if os.path.exists(checkpoint_path):
                            print(f"[Model {model_num}] Checkpoint saved successfully for epoch {epoch+1}")
                        else:
                            logging.warning(f"[Model {model_num}] Checkpoint file not found after saving: {checkpoint_path}")
                    except Exception as e:
                        logging.error(f"[Model {model_num}] Failed to save checkpoint at epoch {epoch+1}: {e}")
                        print(f"[Model {model_num}] Failed to save checkpoint at epoch {epoch+1}: {e}")
                
                if val_loader and early_stopping(avg_val_loss, model):
                    print(f"[Model {model_num}] Early stopping triggered at epoch {epoch+1}")
                    break
                
                print(f'[Model {model_num}] Epoch {epoch+1}/{epochs}: Final Train Loss: {avg_train_loss:.4f}, Final Train Acc: {train_accuracy:.4f} Final Val Loss: {avg_val_loss:.4f}, Final Val Acc: {val_accuracy:.4f}')
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"[Model {model_num}] Error in epoch {epoch+1}: {str(e)}")
                logging.error(f"[Model {model_num}] Epoch {epoch+1} failed: {str(e)}")
                continue
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"model_{model_num}_")]
        print(f"[Model {model_num}] Saved checkpoints: {checkpoint_files}")
        
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
        (model_id, bootstrap_indices, model_class, model_kwargs, epochs, batch_size, 
        learning_rate, save_models, result_dir, train_dataset_info, 
        val_dataset_info, random_seed, num_workers, gpu_id, checkpoint_dir) = args
        
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
        
        print(f"[Model {model_id}] Initializing model {model_class.__name__}...")
        model = model_class(**model_kwargs)
        model = model.to(device)
        
        print(f"[Model {model_id}] Starting training...")
        checkpoint_subdir = os.path.join(checkpoint_dir, f"model_{model_id}_checkpoints")
        try:
            os.makedirs(checkpoint_subdir, exist_ok=True)
            logging.info(f"[Model {model_id}] Created checkpoint subdirectory: {checkpoint_subdir}")
        except Exception as e:
            logging.error(f"[Model {model_id}] Failed to create checkpoint subdirectory {checkpoint_subdir}: {e}")
            raise
        
        log_file = os.path.join(checkpoint_subdir, f"model_{model_id}_checkpoint.log")
        start_time = time.time()
        history = _safe_train_single_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            model_num=model_id,
            device=device,
            checkpoint_dir=checkpoint_subdir,
            save_models=save_models,
            log_file=log_file,
        )
        
        training_time = time.time() - start_time
        final_val_acc = 0.0
        if history and 'val_acc' in history and history['val_acc']:
            final_val_acc = max(history['val_acc'])
        
        if history and save_models:
            performance_file = os.path.join(result_dir, f"model_{model_id}_performance.json")
            os.makedirs(result_dir, exist_ok=True)
            performance_metrics = {
                'model_id': model_id,
                'model_class': model_class.__name__,
                'final_val_acc': final_val_acc,
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'final_train_acc': history['train_acc'][-1] if history['train_acc'] else None,
                'training_time': training_time
            }
            try:
                with open(performance_file, 'w') as f:
                    json.dump(performance_metrics, f, indent=2)
                print(f"[Model {model_id}] Performance metrics saved to {performance_file}")
            except Exception as e:
                logging.error(f"[Model {model_id}] Failed to save performance metrics: {e}")
        
        model_state = model.state_dict() if save_models else None
        
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
        
        print(f"[Model {model_id}] Training completed successfully! Final val acc: {final_val_acc:.4f}, Training time: {training_time:.2f}s")
        
        return {
            'model_id': model_id,
            'model_class': model_class.__name__,
            'history': history,
            'model_state': model_state,
            'final_val_acc': final_val_acc,
            'training_time': training_time,
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
            'model_class': None,
            'history': None,
            'model_state': None,
            'final_val_acc': 0.0,
            'training_time': 0.0,
            'success': False,
            'error': error_msg
        }

def evaluate_single_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    prediction_time = time.time() - start_time
    
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
        'targets': all_targets,
        'prediction_time': prediction_time
    }
    
    return metrics

class BaggingEnsemble:
    def __init__(self, base_model_classes, model_kwargs, n_estimators=10, subsample_ratio=0.8, 
                 bootstrap=True, random_state=42, max_workers=None, use_threading=False, checkpoint_dir=None):
        self.base_model_classes = base_model_classes
        self.model_kwargs = model_kwargs
        self.n_estimators = n_estimators
        self.subsample_ratio = subsample_ratio
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.checkpoint_dir = checkpoint_dir
        self.use_threading = use_threading
        
        if max_workers is None:
            if use_threading:
                self.max_workers = min(n_estimators, 2)
            else:
                self.max_workers = min(n_estimators, max(1, mp.cpu_count() // 2))
        else:
            self.max_workers = max_workers
        
        self.models = []
        self.model_states = []
        self.model_performances = []
        self.best_model_id = None
        self.total_training_time = 0.0
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
        print(f"  - Model classes: {[cls.__name__ for cls in base_model_classes]}")
        
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
            learning_rate=0.001, save_models=True, result_dir="bagging_models", checkpoint_dir=None):
        if save_models:
            try:
                os.makedirs(result_dir, exist_ok=True)
                logging.info(f"Created result directory: {result_dir}")
            except Exception as e:
                logging.error(f"Failed to create result directory {result_dir}: {e}")
                raise
        
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
            model_class = self.base_model_classes[i % len(self.base_model_classes)]
            
            model_checkpoint_dir = os.path.join(self.checkpoint_dir, f"model_{i+1}_checkpoints")
            
            args = (
                i + 1,
                bootstrap_indices,
                model_class,
                self.model_kwargs,
                epochs,
                batch_size,
                learning_rate,
                save_models,
                result_dir,
                train_dataset_info,
                val_dataset_info,
                self.random_state,
                4,
                gpu_id,
                model_checkpoint_dir,
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
                        if result['model_state']:
                            self.model_states.append(result['model_state'])
                        
                        completed_models += 1
                        elapsed_time = time.time() - start_time
                        print(f"\n✓ Model {model_id} ({result['model_class']}) completed! "
                            f"Val Acc: {result['final_val_acc']:.4f} "
                            f"Training Time: {result['training_time']:.2f}s "
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
        
        self.total_training_time = time.time() - start_time
        
        if model_results:
            best_result = max(model_results, key=lambda x: x['final_val_acc'])
            self.best_model_id = best_result['model_id']
            
            print(f"\n🏆 Best Model: Model {self.best_model_id} ({best_result['model_class']}) with validation accuracy: {best_result['final_val_acc']:.4f}")
            
            self.model_performances = [
                {
                    'model_id': result['model_id'],
                    'model_class': result['model_class'],
                    'final_val_acc': result['final_val_acc'],
                    'final_train_loss': result['history']['train_loss'][-1] if result['history'] and result['history']['train_loss'] else None,
                    'final_val_loss': result['history']['val_loss'][-1] if result['history'] and result['history']['val_loss'] else None,
                    'final_train_acc': result['history']['train_acc'][-1] if result['history'] and result['history']['train_acc'] else None,
                    'training_time': result['training_time']
                }
                for result in model_results
            ]
            
            self.model_performances.sort(key=lambda x: x['final_val_acc'], reverse=True)
            
            if save_models:
                ensemble_state_dict = {
                    f'model_{i+1}': state for i, state in enumerate(self.model_states)
                }
                ensemble_state_dict['model_classes'] = [cls.__name__ for cls in self.base_model_classes]
                ensemble_state_dict['model_performances'] = self.model_performances
                ensemble_state_dict['best_model_id'] = self.best_model_id
                ensemble_state_dict['total_training_time'] = self.total_training_time
                
                ensemble_save_path = os.path.join(checkpoint_dir, "ensemble.pth")
                try:
                    torch.save(ensemble_state_dict, ensemble_save_path)
                    print(f"Ensemble state saved to: {ensemble_save_path}")
                except Exception as e:
                    logging.error(f"Failed to save ensemble state to {ensemble_save_path}: {e}")
                    print(f"Failed to save ensemble state to {ensemble_save_path}: {e}")
            
            performance_file = os.path.join(result_dir, "model_performance_summary.json")
            try:
                os.makedirs(result_dir, exist_ok=True)
                with open(performance_file, 'w') as f:
                    json.dump({
                        'best_model_id': self.best_model_id,
                        'model_performances': self.model_performances,
                        'total_training_time': self.total_training_time
                    }, f, indent=2)
                print(f"Model performance summary saved to: {performance_file}")
            except Exception as e:
                logging.error(f"Failed to save performance summary to {performance_file}: {e}")
                print(f"Failed to save performance summary to {performance_file}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total training time: {self.total_training_time:.2f} seconds")
        print(f"Successfully trained: {completed_models}/{self.n_estimators} models")
        print(f"Failed models: {failed_models}")
        print(f"{'='*60}")
        
        self._load_trained_models()
        
        return [h for h in training_history if h is not None]
    
    def _load_trained_models(self):
        self.models = []
        
        ensemble_save_path = os.path.join(self.checkpoint_dir, "ensemble.pth")
        if os.path.exists(ensemble_save_path):
            try:
                ensemble_state_dict = torch.load(ensemble_save_path, map_location=self.device)
                self.model_performances = ensemble_state_dict.get('model_performances', [])
                self.best_model_id = ensemble_state_dict.get('best_model_id')
                self.total_training_time = ensemble_state_dict.get('total_training_time', 0.0)
                
                for i in range(self.n_estimators):
                    model_class = self.base_model_classes[i % len(self.base_model_classes)]
                    model = model_class(**self.model_kwargs)
                    model_state = ensemble_state_dict.get(f'model_{i+1}')
                    if model_state:
                        model.load_state_dict(model_state)
                        model.to(self.device)
                        self.models.append(model)
                        print(f"✓ Loaded model {i+1} ({model_class.__name__}) from ensemble.pth")
                    else:
                        print(f"✗ Model state for model {i+1} not found in ensemble.pth")
                        logging.error(f"Model state for model {i+1} not found in ensemble.pth")
            except Exception as e:
                print(f"✗ Failed to load ensemble from {ensemble_save_path}: {str(e)}")
                logging.error(f"Failed to load ensemble from {ensemble_save_path}: {str(e)}")
        else:
            print(f"✗ Ensemble file not found: {ensemble_save_path}")
            logging.error(f"Ensemble file not found: {ensemble_save_path}")
    
    def get_best_model(self):
        if not self.models or not self.best_model_id:
            raise ValueError("Best model not found or not trained")
        
        best_model_index = next(i for i, p in enumerate(self.model_performances) if p['model_id'] == self.best_model_id)
        return self.models[best_model_index]
    
    def evaluate_individual_models(self, test_loader, class_names=None):
        if not self.models:
            raise ValueError("No trained models found. Please train the ensemble first.")
        
        print(f"\n{'='*60}")
        print("EVALUATING INDIVIDUAL MODELS")
        print(f"{'='*60}")
        
        individual_results = []
        
        for i, model in enumerate(self.models):
            model_class = self.base_model_classes[i % len(self.base_model_classes)]
            print(f"\nEvaluating Model {i+1} ({model_class.__name__})...")
            metrics = evaluate_single_model(model, test_loader, self.device)
            
            result = {
                'model_id': i + 1,
                'model_class': model_class.__name__,
                'metrics': metrics
            }
            individual_results.append(result)
            
            print(f"Model {i+1} ({model_class.__name__}) Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 (Macro): {metrics['f1_weighted']:.4f}")
            print(f"  Prediction Time: {metrics['prediction_time']:.2f}s")
            
            if class_names:
                print(f"Classification Report for Model {i+1}:")
                print(classification_report(metrics['targets'], metrics['predictions'], target_names=class_names))
        
        best_test_result = max(individual_results, key=lambda x: x['metrics']['f1_macro'])
        
        print(f"\n🏆 BEST MODEL ON TEST SET:")
        print(f"Model {best_test_result['model_id']} ({best_test_result['model_class']}) with F1 macro: {best_test_result['metrics']['f1_macro']:.4f}")
        
        return individual_results
    
    def predict(self, test_loader, voting='soft'):
        if not self.models:
            raise ValueError("No trained models found. Please train the ensemble first.")
        
        print(f"Making predictions with {len(self.models)} models...")
        
        start_time = time.time()
        all_predictions = []
        all_probabilities = []
        
        for i, model in enumerate(self.models):
            model_class = self.base_model_classes[i % len(self.base_model_classes)]
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
            print(f"✓ Model {i+1} ({model_class.__name__}) predictions completed")
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        if voting == 'soft':
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        else:
            ensemble_predictions = []
            for i in range(all_predictions.shape[1]):
                votes = all_predictions[:, i]
                ensemble_predictions.append(np.bincount(votes).argmax())
            ensemble_probabilities = None
        
        prediction_time = time.time() - start_time
        print(f"Ensemble prediction completed in {prediction_time:.2f} seconds")
        
        return ensemble_predictions, all_predictions, ensemble_probabilities, prediction_time
    
    def evaluate(self, test_loader, true_labels=None, class_names=None, save_dir="bagging_training_result"):
        if not self.models:
            raise ValueError("No trained models found. Please train the ensemble first.")
        
        print(f"\n{'='*60}")
        print("EVALUATING ENSEMBLE")
        print(f"{'='*60}")
        
        individual_results = self.evaluate_individual_models(test_loader, class_names)
        
        predictions, individual_preds, probabilities, prediction_time = self.predict(test_loader, voting='soft')
        
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
            'precision_weighted': np.mean([r['metrics']['precision_weighted'] for r in individual_results]),
            'prediction_time': np.mean([r['metrics']['prediction_time'] for r in individual_results])
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
        print(f"  Total Training Time: {self.total_training_time:.2f}s")
        print(f"  Total Prediction Time: {prediction_time:.2f}s")
        
        print(f"\nAverage Individual Model Metrics (for reference):")
        print(f"{'='*50}")
        print(f"  Accuracy: {individual_metrics['accuracy']:.4f}")
        print(f"  F1 (Macro): {individual_metrics['f1_macro']:.4f}")
        print(f"  Average Prediction Time: {individual_metrics['prediction_time']:.2f}s")
        
        if class_names:
            print(f"\nClassification Report for Ensemble:")
            print(classification_report(true_labels, predictions, target_names=class_names))
        
        if class_names:
            print("\nGenerating and saving Confusion Matrix for Ensemble...")
            try:
                cm = confusion_matrix(true_labels, predictions)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=class_names, yticklabels=class_names)
                plt.title('Confusion Matrix for Ensemble')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                cm_save_path = os.path.join(save_dir, 'ensemble_confusion_matrix.png')
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Confusion matrix saved to: {cm_save_path}")
            except Exception as e:
                logging.error(f"Failed to save confusion matrix: {e}")
                print(f"Failed to save confusion matrix: {e}")
        
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
                'precision_weighted': ensemble_precision_weighted,
                'total_training_time': self.total_training_time,
                'total_prediction_time': prediction_time
            },
            'individual_metrics': individual_metrics,
            'individual_results': individual_results,
            'ensemble_predictions': predictions.tolist(),
            'individual_predictions': individual_preds.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'confusion_matrix': cm.tolist() if class_names else None,
            'class_names': class_names
        }
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            metrics_path = os.path.join(save_dir, "ensemble_evaluation_results.json")
            with open(metrics_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved ensemble evaluation results to {metrics_path}")
            
            predictions_path = os.path.join(save_dir, "ensemble_predictions.json")
            with open(predictions_path, 'w') as f:
                json.dump({
                    'ensemble_predictions': predictions.tolist(),
                    'individual_predictions': individual_preds.tolist(),
                    'probabilities': probabilities.tolist() if probabilities is not None else None
                }, f, indent=2)
            print(f"Saved predictions to {predictions_path}")
        except Exception as e:
            logging.error(f"Failed to save evaluation results or predictions to {save_dir}: {e}")
            print(f"Failed to save evaluation results or predictions to {save_dir}: {e}")
        
        return results
    
    def save_ensemble(self, save_dir="bagging_ensemble"):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create save directory {save_dir}: {e}")
            raise
        
        ensemble_state = {
            'n_estimators': self.n_estimators,
            'subsample_ratio': self.subsample_ratio,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'model_kwargs': self.model_kwargs,
            'model_classes': [cls.__name__ for cls in self.base_model_classes],
            'model_performances': self.model_performances,
            'best_model_id': self.best_model_id,
            'total_training_time': self.total_training_time
        }
        
        state_file = os.path.join(save_dir, "ensemble_state.json")
        try:
            with open(state_file, 'w') as f:
                json.dump(ensemble_state, f, indent=2)
            print(f"Ensemble state saved to: {state_file}")
        except Exception as e:
            logging.error(f"Failed to save ensemble state to {state_file}: {e}")
            print(f"Failed to save ensemble state to {state_file}: {e}")

    def load_ensemble(self, save_dir="bagging_ensemble"):
        state_file = os.path.join(save_dir, "ensemble_state.json")
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"Ensemble state file not found: {state_file}")
        
        try:
            with open(state_file, 'r') as f:
                ensemble_state = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load ensemble state from {state_file}: {e}")
            raise
        
        self.n_estimators = ensemble_state['n_estimators']
        self.subsample_ratio = ensemble_state['subsample_ratio']
        self.bootstrap = ensemble_state['bootstrap']
        self.random_state = ensemble_state['random_state']
        self.model_kwargs = ensemble_state['model_kwargs']
        self.model_performances = ensemble_state['model_performances']
        self.best_model_id = ensemble_state['best_model_id']
        self.total_training_time = ensemble_state.get('total_training_time', 0.0)
        
        model_registry = get_model_registry()
        self.base_model_classes = [model_registry[name] for name in ensemble_state['model_classes']]
        
        self._load_trained_models()
        
        print(f"Ensemble loaded from: {state_file}")

    def plot_training_history(self, history_list, save_dir="training_history_plots", dpi=300, image_format="png"):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create save directory {save_dir}: {e}")
            raise
        
        for i, history in enumerate(history_list):
            if history is None:
                print(f"Skipping Model {i+1}: No history available")
                continue
            
            model_class = self.base_model_classes[i % len(self.base_model_classes)]
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_acc'], label='train', color='blue', linewidth=2)
            if history['val_acc']:
                plt.plot(history['val_acc'], '--', label='valid', color='orange', linewidth=2)
            plt.title(f'Model {i+1} ({model_class.__name__}) Accuracy')
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
            plt.title(f'Model {i+1} ({model_class.__name__}) Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim(0, max(max(history['train_loss'], default=2), max(history['val_loss'], default=2)) * 1.1)
            plt.xticks(range(0, len(history['train_loss']), max(1, len(history['train_loss'])//5)))
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"model_{i+1}_history.{image_format}")
            try:
                plt.savefig(save_path, format=image_format, dpi=dpi, bbox_inches='tight')
                print(f"Training history plot for Model {i+1} ({model_class.__name__}) saved to: {save_path}")
            except Exception as e:
                logging.error(f"Failed to save training history plot for Model {i+1}: {e}")
                print(f"Failed to save training history plot for Model {i+1}: {e}")
            plt.close()

    def plot_model_performance(self, save_dir="model_performance_plots", dpi=300, image_format="png"):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create save directory {save_dir}: {e}")
            raise
        
        model_ids = [p['model_id'] for p in self.model_performances]
        model_classes = [p['model_class'] for p in self.model_performances]
        val_accs = [p['final_val_acc'] for p in self.model_performances]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(model_ids)), val_accs, color='skyblue')
        plt.xlabel('Model ID (Class)')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy of Individual Models in Ensemble')
        plt.xticks(range(len(model_ids)), [f"{id} ({cls})" for id, cls in zip(model_ids, model_classes)], rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path = os.path.join(save_dir, f"model_performance.{image_format}")
        try:
            plt.savefig(save_path, format=image_format, dpi=dpi, bbox_inches='tight')
            print(f"Model performance plot saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save model performance plot: {e}")
            print(f"Failed to save model performance plot: {e}")
        plt.close()

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
    return {
        'efficientnetv2_fastkan': EfficientNetV2FastKAN,
    }

def save_training_config(config_path, config):
    serializable_config = config.copy()
    
    if 'base_model_classes' in serializable_config:
        serializable_config['base_model_classes'] = [cls.__name__ for cls in serializable_config['base_model_classes']]
    
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
        logging.info(f"Saved training config to {config_path}")
    except (IOError, PermissionError) as e:
        logging.error(f"Failed to save config to {config_path}: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a bagging ensemble with multiple FastKAN models using a zip file dataset.")
    parser.add_argument('--data-dir', type=str, default="data.zip", help='Path to the zip file containing the dataset.')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the training configuration file.')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate the existing ensemble.pth without training.')
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
        'result_dir': 'bagging_training_results/ensemble_efficientnetv2',
        'plot_save_dir': 'training_history_plots/ensemble_efficientnetv2',
        'data_dir': 'data.zip',
        'checkpoint_dir': 'bagging_checkpoints/ensemble_efficientnetv2',
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
            {'name': 'efficientnetv2_fastkan', 'class': EfficientNetV2FastKAN
            }
        ]
    }
    
    try:
        os.makedirs(default_config['result_dir'], exist_ok=True)
        os.makedirs(default_config['plot_save_dir'], exist_ok=True)
        os.makedirs(default_config['checkpoint_dir'], exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create default directories: {e}")
        raise
    
    model_registry = get_model_registry()
    print("Available models:")
    for model_name, model_class in model_registry.items():
        print(f"  - {model_name}: {model_class.__name__}")
    
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
    
    if args.data_dir and os.path.exists(args.data_dir):
        config['data_dir'] = args.data_dir
    
    if not os.path.exists(config['data_dir']):
        raise FileNotFoundError(f"Data zip file not found at {config['data_dir']}")
    
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
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ZipImageDataset(zip_path=config['data_dir'], subset='train', transform=train_transform)
    val_dataset = ZipImageDataset(zip_path=config['data_dir'], subset='val', transform=val_transform)
    test_dataset = ZipImageDataset(zip_path=config['data_dir'], subset='test', transform=test_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(f"No images found in {config['data_dir']} for {'train' if len(train_dataset) == 0 else 'val' if len(val_dataset) == 0 else 'test'} subset")
    
    num_classes = len(train_dataset.classes)
    config['model_kwargs']['num_classes'] = num_classes
    
    class_names = train_dataset.classes 
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    print("\n=== Training or Evaluating Ensemble ===")
    
    model_classes = [model['class'] for model in config['models'][:10]]
    
    save_config_path = os.path.join(config['result_dir'], "ensemble_training_config.json")
    config['base_model_classes'] = model_classes
    try:
        save_training_config(save_config_path, config)
        logging.info(f"Saved ensemble config to {save_config_path}")
        print(f"Saved ensemble config to {save_config_path}")
    except Exception as e:
        logging.error(f"Failed to save ensemble config: {e}")
        raise
    
    try:
        ensemble = BaggingEnsemble(
            base_model_classes=model_classes,
            model_kwargs=config['model_kwargs'],
            n_estimators=config['n_estimators'],
            subsample_ratio=config['subsample_ratio'],
            bootstrap=True,
            checkpoint_dir=config['checkpoint_dir'],
            random_state=config['random_state'],
            max_workers=config['max_workers'],
            use_threading=config['use_threading'],
        )
        
        if not args.evaluate_only:
            print("\n=== Training Ensemble ===")
            training_history = ensemble.fit(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                save_models=not config['no_save_models'],
                result_dir=config['result_dir'],
                checkpoint_dir=config['checkpoint_dir'],
            )
            
            ensemble.save_ensemble(save_dir=config['result_dir'])
            
            if training_history:
                try:
                    ensemble.plot_training_history(
                        training_history,
                        save_dir=config['plot_save_dir'],
                        dpi=config['dpi'],
                        image_format=config['image_format']
                    )
                    ensemble.plot_model_performance(
                        save_dir=config['plot_save_dir'],
                        dpi=config['dpi'],
                        image_format=config['image_format']
                    )
                except Exception as e:
                    logging.error(f"Failed to plot training history or model performance: {e}")
                    print(f"Failed to plot training history or model performance: {e}")
        else:
            print("\n=== Loading Existing Ensemble ===")
            ensemble.load_ensemble(save_dir=config['result_dir'])
        
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
        
        print("\n=== Evaluating Ensemble ===")
        results = ensemble.evaluate(
            test_loader=test_loader,
            class_names=class_names,
            save_dir=config['result_dir']
        )
        
        print(f"\nEnsemble Evaluation Results:")
        print(f"Accuracy: {results['ensemble_metrics']['accuracy']:.4f}")
        print(f"F1 (Macro): {results['ensemble_metrics']['f1_macro']:.4f}")
        print(f"Total Training Time: {results['ensemble_metrics']['total_training_time']:.2f}s")
        print(f"Total Prediction Time: {results['ensemble_metrics']['total_prediction_time']:.2f}s")
        print(f"Confusion Matrix saved to: {os.path.join(config['result_dir'], 'ensemble_confusion_matrix.png')}")
        print(f"Evaluation results saved to: {os.path.join(config['result_dir'], 'ensemble_evaluation_results.json')}")
        print(f"Predictions saved to: {os.path.join(config['result_dir'], 'ensemble_predictions.json')}")
        
        ensemble.cleanup()
        
    except Exception as e:
        logging.error(f"Training or evaluation failed for ensemble: {str(e)}")
        print(f"Training or evaluation failed for ensemble: {str(e)}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nEnsemble training and evaluation completed!")

if __name__ == "__main__":
    main()