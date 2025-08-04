"""
Meta-Composer: Lightweight neural model for primitive sequence prediction
Combines transformer architecture with task-specific heads for ultra-compact solutions
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Meta-composer will not work.")

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import math

if TORCH_AVAILABLE:
    from .tokenizer import ARCTokenizer, FeatureExtractor
else:
    # Stub classes when torch is not available
    class ARCTokenizer:
        def __init__(self, *args, **kwargs):
            pass
    class FeatureExtractor:
        def __init__(self, *args, **kwargs):
            pass

if not TORCH_AVAILABLE:
    # Stub classes when torch is not available
    class PositionalEncoding:
        def __init__(self, *args, **kwargs): pass
    class MicroTransformer:
        def __init__(self, *args, **kwargs): pass
    class ARCDataset:
        def __init__(self, *args, **kwargs): pass
    class MetaComposer:
        def __init__(self, *args, **kwargs): pass
    def create_primitive_vocab(): return {}
else:

    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MicroTransformer(nn.Module):
    """Ultra-lightweight transformer for primitive sequence prediction"""
    
    def __init__(self, 
                 vocab_size: int = 256,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 max_seq_len: int = 100,
                 num_primitives: int = 20):
        super().__init__()
        
        self.d_model = d_model
        self.num_primitives = num_primitives
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.primitive_head = nn.Linear(d_model, num_primitives)  # Primitive selection
        self.sequence_head = nn.Linear(d_model, 10)  # Sequence length prediction
        self.parameter_head = nn.Linear(d_model, 32)  # Parameter prediction
        
        # Feature fusion layer
        self.feature_projection = nn.Linear(64, d_model)  # For auxiliary features
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, 
                tokens: torch.Tensor, 
                features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            tokens: [batch_size, seq_len] - tokenized examples
            features: [batch_size, feature_dim] - auxiliary features
            attention_mask: [batch_size, seq_len] - attention mask
            
        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len = tokens.shape
        
        # Token embedding and positional encoding
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Add feature information if available
        if features is not None:
            feature_emb = self.feature_projection(features)  # [batch_size, d_model]
            feature_emb = feature_emb.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + feature_emb
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to transformer format
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # Invert mask
        
        transformer_out = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Global pooling for sequence-level predictions
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = (~attention_mask).unsqueeze(-1).expand_as(transformer_out)
            masked_out = transformer_out * mask_expanded
            pooled = masked_out.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = transformer_out.mean(dim=1)
        
        # Multiple prediction heads
        primitive_logits = self.primitive_head(pooled)  # [batch_size, num_primitives]
        sequence_length = self.sequence_head(pooled)     # [batch_size, 10]
        parameters = self.parameter_head(pooled)         # [batch_size, 32]
        
        return {
            'primitive_logits': primitive_logits,
            'sequence_length': sequence_length,
            'parameters': parameters,
            'hidden_states': transformer_out
        }
    
    def predict_primitive_sequence(self, 
                                 tokens: torch.Tensor,
                                 features: Optional[torch.Tensor] = None,
                                 max_length: int = 5) -> List[str]:
        """Predict sequence of primitives for given input"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(tokens, features)
            
            # Get primitive probabilities
            primitive_probs = F.softmax(outputs['primitive_logits'], dim=-1)
            
            # Get sequence length
            seq_len_logits = outputs['sequence_length']
            predicted_length = torch.argmax(seq_len_logits, dim=-1).item() + 1
            predicted_length = min(predicted_length, max_length)
            
            # Select top primitives
            top_primitives = torch.topk(primitive_probs, predicted_length, dim=-1)
            
            # Convert to primitive names (assuming we have a mapping)
            primitive_names = [f'prim_{idx.item()}' for idx in top_primitives.indices[0]]
            
            return primitive_names[:predicted_length]


class ARCDataset(Dataset):
    """Dataset for ARC task examples with primitive sequence labels"""
    
    def __init__(self, 
                 tasks: List[Dict],
                 tokenizer: ARCTokenizer,
                 feature_extractor: FeatureExtractor,
                 primitive_vocab: Dict[str, int]):
        
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.primitive_vocab = primitive_vocab
        self.data = self._prepare_data(tasks)
    
    def _prepare_data(self, tasks: List[Dict]) -> List[Dict]:
        """Prepare training data from tasks"""
        prepared = []
        
        for task in tasks:
            examples = task.get('train', [])
            if not examples:
                continue
            
            # Tokenize examples
            tokens = self.tokenizer.tokenize_examples(examples)
            
            # Extract features
            features = self.feature_extractor.extract_features(examples)
            
            # Get ground truth primitive sequence (if available)
            primitive_sequence = task.get('primitive_sequence', ['mc', 'r90'])  # Default
            
            # Convert primitive names to indices
            primitive_indices = [
                self.primitive_vocab.get(prim, 0) 
                for prim in primitive_sequence
            ]
            
            prepared.append({
                'tokens': torch.tensor(tokens, dtype=torch.long),
                'features': torch.tensor(features, dtype=torch.float32),
                'primitive_sequence': primitive_indices,
                'sequence_length': len(primitive_sequence),
                'task_id': task.get('id', 'unknown')
            })
        
        return prepared
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MetaComposer:
    """Main meta-learning system for primitive sequence prediction"""
    
    def __init__(self, 
                 primitive_vocab: Dict[str, int],
                 model_config: Optional[Dict] = None):
        
        self.primitive_vocab = primitive_vocab
        self.reverse_vocab = {v: k for k, v in primitive_vocab.items()}
        
        # Initialize components
        self.tokenizer = ARCTokenizer(max_tokens=100)
        self.feature_extractor = FeatureExtractor()
        
        # Model configuration
        config = model_config or {}
        self.model = MicroTransformer(
            vocab_size=config.get('vocab_size', 256),
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            num_primitives=len(primitive_vocab)
        )
        
        # Training components
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training history
        self.training_history = []
    
    def prepare_training(self, 
                        learning_rate: float = 1e-3,
                        weight_decay: float = 1e-4):
        """Prepare model for training"""
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Multi-task loss
        self.primitive_criterion = nn.CrossEntropyLoss()
        self.sequence_criterion = nn.CrossEntropyLoss()
        self.parameter_criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        primitive_loss = 0.0
        sequence_loss = 0.0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Move to device
            tokens = batch['tokens'].to(self.device)
            features = batch['features'].to(self.device)
            
            # Create attention mask (ignore padding tokens)
            attention_mask = (tokens != 0).float()
            
            # Forward pass
            outputs = self.model(tokens, features, attention_mask)
            
            # Compute losses
            batch_size = tokens.shape[0]
            
            # Primitive prediction loss (use first primitive as target)
            if 'primitive_sequence' in batch:
                first_primitives = torch.tensor([
                    seq[0] if seq else 0 
                    for seq in batch['primitive_sequence']
                ]).to(self.device)
                
                prim_loss = self.primitive_criterion(
                    outputs['primitive_logits'], 
                    first_primitives
                )
                primitive_loss += prim_loss.item()
            else:
                prim_loss = 0
            
            # Sequence length loss
            if 'sequence_length' in batch:
                seq_lengths = torch.tensor([
                    min(length - 1, 9)  # Clamp to 0-9 range
                    for length in batch['sequence_length']
                ]).to(self.device)
                
                seq_loss = self.sequence_criterion(
                    outputs['sequence_length'],
                    seq_lengths
                )
                sequence_loss += seq_loss.item()
            else:
                seq_loss = 0
            
            # Total loss
            loss = prim_loss + 0.5 * seq_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'primitive_loss': primitive_loss / len(dataloader),
            'sequence_loss': sequence_loss / len(dataloader)
        }
    
    def predict_sequence(self, examples: List[Dict]) -> List[str]:
        """Predict primitive sequence for given examples"""
        
        # Tokenize and extract features
        tokens = torch.tensor(
            self.tokenizer.tokenize_examples(examples), 
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        features = torch.tensor(
            self.feature_extractor.extract_features(examples),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tokens, features)
            
            # Get top primitive predictions
            primitive_probs = F.softmax(outputs['primitive_logits'], dim=-1)
            top_k = min(5, len(self.primitive_vocab))
            top_primitives = torch.topk(primitive_probs, top_k, dim=-1)
            
            # Convert indices to primitive names
            sequence = []
            for idx in top_primitives.indices[0]:
                prim_name = self.reverse_vocab.get(idx.item(), 'unknown')
                if prim_name != 'unknown':
                    sequence.append(prim_name)
            
            # Predict sequence length
            seq_len_probs = F.softmax(outputs['sequence_length'], dim=-1)
            predicted_length = torch.argmax(seq_len_probs, dim=-1).item() + 1
            
            return sequence[:predicted_length]
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'primitive_vocab': self.primitive_vocab,
            'training_history': self.training_history
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.primitive_vocab = checkpoint.get('primitive_vocab', self.primitive_vocab)
        self.training_history = checkpoint.get('training_history', [])
    
    def estimate_model_size(self) -> int:
        """Estimate model size in parameters"""
        return sum(p.numel() for p in self.model.parameters())


def create_primitive_vocab() -> Dict[str, int]:
    """Create vocabulary mapping for primitives"""
    primitives = [
        'r90', 'fh', 'fv', 'tr', 'sh',  # geometry
        'mc', 'tm', 'rc', 'bc', 'md',   # color ops
        'ff', 'bb', 'ct', 'cc',         # shape ops
        'inc', 'cl', 'he', 'sm', 'avg', # numeric
        'flood_nca', 'edge_nca'         # nca ops
    ]
    
    return {prim: i for i, prim in enumerate(primitives)}


if __name__ == "__main__":
    # Demo usage
    primitive_vocab = create_primitive_vocab()
    
    # Create meta-composer
    composer = MetaComposer(primitive_vocab)
    
    print(f"Model parameters: {composer.estimate_model_size():,}")
    print(f"Primitive vocabulary size: {len(primitive_vocab)}")
    
    # Example prediction
    examples = [{
        'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    }]
    
    predicted_sequence = composer.predict_sequence(examples)
    print(f"Predicted sequence: {predicted_sequence}")
