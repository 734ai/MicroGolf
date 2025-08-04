#!/usr/bin/env python3
"""
Advanced Meta-Controller Training Script for MicroGolf
Trains the lightweight transformer model on ARC tasks with primitive sequences
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from microgolf.model import MetaComposer, ARCDataset, create_primitive_vocab
from microgolf.model.tokenizer import ARCTokenizer, FeatureExtractor
from microgolf.engine import PrimitiveController
from microgolf.data_loader import ARCDataLoader


class TrainingManager:
    """Advanced training manager with validation, checkpointing, and monitoring"""
    
    def __init__(self, 
                 model: MetaComposer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any]):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="microgolf-meta-composer",
                config=config,
                name=f"run_{config.get('run_name', 'default')}"
            )
    
    def train(self, num_epochs: int):
        """Main training loop with validation and checkpointing"""
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Logging
            self._log_metrics(train_metrics, val_metrics)
            
            # Checkpointing
            self._checkpoint(val_metrics['total_loss'])
            
            # Early stopping
            if self._should_stop():
                print(f"Early stopping at epoch {epoch}")
                break
            
            print(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.model.train()
        metrics = self.model.train_epoch(self.train_loader)
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.model.eval()
        total_loss = 0.0
        primitive_accuracy = 0.0
        sequence_accuracy = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                tokens = batch['tokens'].to(self.model.device)
                features = batch['features'].to(self.model.device)
                attention_mask = (tokens != 0).float()
                
                outputs = self.model.model(tokens, features, attention_mask)
                
                # Compute validation loss (simplified)
                if 'primitive_sequence' in batch:
                    first_primitives = torch.tensor([
                        seq[0] if seq else 0 
                        for seq in batch['primitive_sequence']
                    ]).to(self.model.device)
                    
                    prim_loss = self.model.primitive_criterion(
                        outputs['primitive_logits'], 
                        first_primitives
                    )
                    total_loss += prim_loss.item()
                    
                    # Accuracy
                    pred_prims = torch.argmax(outputs['primitive_logits'], dim=-1)
                    primitive_accuracy += (pred_prims == first_primitives).float().mean().item()
        
        return {
            'total_loss': total_loss / len(self.val_loader),
            'primitive_accuracy': primitive_accuracy / len(self.val_loader),
            'sequence_accuracy': sequence_accuracy / len(self.val_loader)
        }
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log training metrics"""
        
        if self.use_wandb:
            wandb.log({
                'epoch': self.epoch,
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
                'primitive_accuracy': val_metrics.get('primitive_accuracy', 0),
                'learning_rate': self.model.optimizer.param_groups[0]['lr']
            })
        
        # Store in model history
        self.model.training_history.append({
            'epoch': self.epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })
    
    def _checkpoint(self, val_loss: float):
        """Save checkpoint if validation loss improved"""
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Save best model
            checkpoint_path = Path(self.config['output_dir']) / 'best_model.pt'
            self.model.save_model(str(checkpoint_path))
            
            print(f"New best model saved with val_loss: {val_loss:.4f}")
        else:
            self.patience_counter += 1
    
    def _should_stop(self) -> bool:
        """Check if early stopping criteria met"""
        patience = self.config.get('patience', 10)
        return self.patience_counter >= patience


def load_arc_tasks_with_solutions(data_dir: Path) -> List[Dict]:
    """Load real ARC tasks and map them to primitive solutions"""
    
    # Load real ARC data
    arc_loader = ARCDataLoader(str(data_dir))
    
    # Get training tasks
    training_tasks = arc_loader.get_training_tasks()
    
    # For now, create primitive mappings using heuristics
    # In a real implementation, these would be learned or hand-labeled
    controller = PrimitiveController()
    
    tasks_with_solutions = []
    
    for arc_task in training_tasks[:100]:  # Limit to first 100 tasks for training
        # Convert ARCTask to our format
        task_data = {
            'id': arc_task.task_id,
            'train': arc_task.train_examples,
            'test': arc_task.test_examples
        }
        
        # Use controller to get primitive sequence suggestion
        try:
            # Use first training example to suggest primitives
            if arc_task.train_examples:
                input_grid = arc_task.train_examples[0]['input']
                output_grid = arc_task.train_examples[0]['output']
                
                # Get primitive candidates from controller
                candidates = controller.get_primitive_candidates(input_grid, output_grid)
                
                # Map single characters to full primitive names
                char_to_primitive = {
                    'r': 'r90', 'f': 'fh', 'v': 'fv', 't': 'tr', 's': 'sh',
                    'm': 'mc', 'tm': 'tm', 'rc': 'rc', 'bc': 'bc', 'md': 'md',
                    'ff': 'ff', 'bb': 'bb', 'ct': 'ct', 'cc': 'cc',
                    'inc': 'inc', 'cl': 'cl', 'he': 'he', 'sm': 'sm', 'avg': 'avg'
                }
                
                # Take top primitive candidates as sequence
                primitive_sequence = []
                for prim_char, _ in candidates[:2]:
                    if prim_char in char_to_primitive:
                        primitive_sequence.append(char_to_primitive[prim_char])
                    else:
                        # Try direct mapping or fallback
                        primitive_sequence.append(prim_char if len(prim_char) > 1 else 'mc')
                
                if not primitive_sequence:
                    primitive_sequence = ['mc']  # Default fallback
                
                task_data['primitive_sequence'] = primitive_sequence
                tasks_with_solutions.append(task_data)
                
        except Exception as e:
            print(f"Warning: Failed to process task {arc_task.task_id}: {e}")
            continue
    
    print(f"Processed {len(tasks_with_solutions)} real ARC tasks with primitive mappings")
    
    # Add some synthetic tasks for validation
    synthetic_tasks = create_synthetic_validation_tasks()
    
    return tasks_with_solutions + synthetic_tasks


def create_synthetic_validation_tasks() -> List[Dict]:
    """Create synthetic validation tasks with known solutions"""
    
    synthetic_tasks = []
    
    # Task 1: Simple color inversion
    task1 = {
        'id': 'synthetic_001',
        'train': [
            {
                'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            },
            {
                'input': [[2, 0, 2], [0, 2, 0]],
                'output': [[0, 2, 0], [2, 0, 2]]
            }
        ],
        'test': [
            {
                'input': [[3, 0, 3], [0, 3, 0], [3, 0, 3]],
                'output': [[0, 3, 0], [3, 0, 3], [0, 3, 0]]
            }
        ],
        'primitive_sequence': ['mc']  # Color mapping
    }
    
    # Task 2: Rotation
    task2 = {
        'id': 'synthetic_002',
        'train': [
            {
                'input': [[1, 2], [3, 4]],
                'output': [[3, 1], [4, 2]]
            }
        ],
        'test': [
            {
                'input': [[5, 6], [7, 8]],
                'output': [[7, 5], [8, 6]]
            }
        ],
        'primitive_sequence': ['r90']
    }
    
    # Task 3: Horizontal flip
    task3 = {
        'id': 'synthetic_003',
        'train': [
            {
                'input': [[1, 2, 3], [4, 5, 6]],
                'output': [[4, 5, 6], [1, 2, 3]]
            }
        ],
        'test': [
            {
                'input': [[7, 8, 9], [1, 2, 3]],
                'output': [[1, 2, 3], [7, 8, 9]]
            }
        ],
        'primitive_sequence': ['fh']
    }
    
    return [task1, task2, task3]


def create_data_loaders(tasks: List[Dict], 
                       tokenizer: ARCTokenizer,
                       feature_extractor: FeatureExtractor,
                       primitive_vocab: Dict[str, int],
                       batch_size: int = 16,
                       val_split: float = 0.2) -> tuple:
    """Create training and validation data loaders"""
    
    # Create dataset
    dataset = ARCDataset(tasks, tokenizer, feature_extractor, primitive_vocab)
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for batching variable-length sequences"""
    
    # Stack tensors
    tokens = torch.stack([item['tokens'] for item in batch])
    features = torch.stack([item['features'] for item in batch])
    
    # Handle variable-length primitive sequences
    primitive_sequences = [item['primitive_sequence'] for item in batch]
    sequence_lengths = [item['sequence_length'] for item in batch]
    
    return {
        'tokens': tokens,
        'features': features,
        'primitive_sequence': primitive_sequences,
        'sequence_length': sequence_lengths
    }


def main():
    parser = argparse.ArgumentParser(description='Train Meta-Composer for MicroGolf')
    
    parser.add_argument('--data_dir', type=str, default='data/arc',
                       help='Directory containing ARC tasks')
    parser.add_argument('--output_dir', type=str, default='experiments/meta',
                       help='Output directory for models and logs')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--run_name', type=str, default='default',
                       help='Run name for logging')
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of transformer layers')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    print("Loading real ARC tasks...")
    tasks = load_arc_tasks_with_solutions(Path(args.data_dir))
    print(f"Loaded {len(tasks)} tasks with primitive mappings")
    
    # Initialize components
    primitive_vocab = create_primitive_vocab()
    tokenizer = ARCTokenizer(max_tokens=100)
    feature_extractor = FeatureExtractor()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        tasks, tokenizer, feature_extractor, primitive_vocab,
        batch_size=args.batch_size
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model_config = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'vocab_size': 256
    }
    
    composer = MetaComposer(primitive_vocab, model_config)
    composer.prepare_training(args.learning_rate, args.weight_decay)
    
    print(f"Model parameters: {composer.estimate_model_size():,}")
    
    # Training manager
    training_manager = TrainingManager(composer, train_loader, val_loader, config)
    
    # Train model
    print("Starting training...")
    training_manager.train(args.epochs)
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    composer.save_model(str(final_model_path))
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(composer.training_history, f, indent=2)
    
    print(f"Training completed. Models saved to {output_dir}")
    
    # Evaluate on a few examples
    print("\nEvaluating on sample tasks...")
    for i, task in enumerate(tasks[:3]):
        examples = task['train']
        true_sequence = task['primitive_sequence']
        predicted_sequence = composer.predict_sequence(examples)
        
        print(f"Task {task['id']}:")
        print(f"  True sequence: {true_sequence}")
        print(f"  Predicted sequence: {predicted_sequence}")
        print()


if __name__ == "__main__":
    main()
