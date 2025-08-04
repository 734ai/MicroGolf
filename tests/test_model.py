"""
Test suite for MicroGolf model components
Tests tokenizer, meta-composer, and training pipeline
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from microgolf.model.tokenizer import ARCTokenizer
from microgolf.model.meta_composer import MicroTransformer, MetaComposer


class TestARCTokenizer:
    """Test the ARC task tokenizer"""
    
    def setup_method(self):
        """Set up test tokenizer"""
        self.tokenizer = ARCTokenizer()
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initializes correctly"""
        assert self.tokenizer is not None
        assert hasattr(self.tokenizer, 'color_vocab')
        assert hasattr(self.tokenizer, 'primitive_vocab')
        assert hasattr(self.tokenizer, 'position_vocab')
        
        # Check vocabularies are populated
        assert len(self.tokenizer.color_vocab) > 0
        assert len(self.tokenizer.primitive_vocab) > 0
        assert len(self.tokenizer.position_vocab) > 0
    
    def test_encode_color_values(self):
        """Test color value encoding"""
        # Simple grid with different colors
        grid = [[0, 1, 2], [3, 4, 5]]
        
        encoded = self.tokenizer.encode_color_values(grid)
        
        # Should return list of tokens
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        
        # All tokens should be integers
        assert all(isinstance(token, int) for token in encoded)
        
        # Should handle color values properly
        for row in grid:
            for color in row:
                if color < len(self.tokenizer.color_vocab):
                    assert color in encoded or color + 1 in encoded  # Account for offset
    
    def test_encode_spatial_relations(self):
        """Test spatial relation encoding"""
        grid = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        
        encoded = self.tokenizer.encode_spatial_relations(grid)
        
        # Should return list of tokens
        assert isinstance(encoded, list)
        assert len(encoded) >= 0  # Might be empty for simple patterns
        
        # All tokens should be valid
        for token in encoded:
            assert isinstance(token, int)
            assert token >= 0
    
    def test_encode_shape_features(self):
        """Test shape feature encoding"""
        # Grid with distinct shape
        grid = [
            [0, 1, 0],
            [1, 1, 1], 
            [0, 1, 0]
        ]
        
        encoded = self.tokenizer.encode_shape_features(grid)
        
        # Should return list of tokens
        assert isinstance(encoded, list)
        assert len(encoded) >= 0
        
        # All tokens should be valid integers
        for token in encoded:
            assert isinstance(token, int)
    
    def test_encode_pattern_primitives(self):
        """Test pattern primitive encoding"""
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[3, 1], [4, 2]]  # Rotated
        
        encoded = self.tokenizer.encode_pattern_primitives(input_grid, output_grid)
        
        # Should return list of tokens
        assert isinstance(encoded, list)
        assert len(encoded) >= 0
        
        # Should contain valid primitive tokens
        for token in encoded:
            assert isinstance(token, int)
            assert token >= 0
    
    def test_encode_arc_task(self):
        """Test full ARC task encoding"""
        # Create sample ARC task
        task = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]], 
                    'output': [[0, 1], [1, 0]]
                },
                {
                    'input': [[2, 0], [0, 2]],
                    'output': [[0, 2], [2, 0]]
                }
            ],
            'test': [
                {
                    'input': [[3, 0], [0, 3]],
                    'output': None  # To be predicted
                }
            ]
        }
        
        encoded = self.tokenizer.encode_arc_task(task)
        
        # Should return dictionary with encoded components
        assert isinstance(encoded, dict)
        assert 'input_tokens' in encoded
        assert 'output_tokens' in encoded
        
        # Tokens should be lists of integers
        assert isinstance(encoded['input_tokens'], list)
        assert isinstance(encoded['output_tokens'], list)
        
        # Should have reasonable length
        assert len(encoded['input_tokens']) > 0
    
    def test_decode_to_primitives(self):
        """Test decoding tokens back to primitives"""
        # Create some test tokens
        test_tokens = [1, 5, 10, 15]  # Arbitrary valid tokens
        
        primitives = self.tokenizer.decode_to_primitives(test_tokens)
        
        # Should return list of primitive names
        assert isinstance(primitives, list)
        
        # All elements should be strings (primitive names)
        for prim in primitives:
            assert isinstance(prim, str)
    
    def test_vocabulary_consistency(self):
        """Test that vocabularies are consistent"""
        # Color vocab should cover common ARC colors (0-9)
        for color in range(10):
            assert color in self.tokenizer.color_vocab
        
        # Primitive vocab should contain known primitives
        known_primitives = ['r90', 'fh', 'fv', 'tr', 'mc', 'rc']
        for prim in known_primitives:
            assert prim in self.tokenizer.primitive_vocab
        
        # Position vocab should have reasonable size
        assert len(self.tokenizer.position_vocab) >= 10
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        empty_grid = []
        
        # Should handle gracefully without crashing
        color_encoded = self.tokenizer.encode_color_values(empty_grid)
        spatial_encoded = self.tokenizer.encode_spatial_relations(empty_grid)
        shape_encoded = self.tokenizer.encode_shape_features(empty_grid)
        
        assert isinstance(color_encoded, list)
        assert isinstance(spatial_encoded, list)
        assert isinstance(shape_encoded, list)
    
    def test_large_input_handling(self):
        """Test handling of large inputs"""
        # Create large grid
        large_grid = [[i % 10 for i in range(30)] for _ in range(30)]
        
        # Should handle without excessive memory usage
        encoded = self.tokenizer.encode_color_values(large_grid)
        
        assert isinstance(encoded, list)
        assert len(encoded) <= 1000  # Should be reasonably compact


class TestMicroTransformer:
    """Test the lightweight transformer model"""
    
    def setup_method(self):
        """Set up test transformer"""
        try:
            self.transformer = MicroTransformer(
                vocab_size=100,
                d_model=32,
                nhead=4,
                num_layers=2,
                max_length=64
            )
        except ImportError:
            self.transformer = None
    
    def test_transformer_initialization(self):
        """Test transformer initializes correctly"""
        if self.transformer is None:
            pytest.skip("PyTorch not available")
        
        assert self.transformer is not None
        assert hasattr(self.transformer, 'embedding')
        assert hasattr(self.transformer, 'pos_encoding')
        assert hasattr(self.transformer, 'transformer')
        assert hasattr(self.transformer, 'output_projection')
        
        # Check parameter count is small
        total_params = sum(p.numel() for p in self.transformer.parameters())
        assert total_params < 50000  # Should be under 50K parameters
    
    def test_transformer_forward(self):
        """Test transformer forward pass"""
        if self.transformer is None:
            pytest.skip("PyTorch not available")
        
        try:
            import torch
            
            # Create test input (batch_size=2, seq_len=10)
            test_input = torch.randint(0, 100, (2, 10))
            
            output = self.transformer(test_input)
            
            # Should return tensor of correct shape
            assert output.shape == (2, 10, 100)  # (batch, seq, vocab)
            
            # Output should be valid logits
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_positional_encoding(self):
        """Test positional encoding component"""
        if self.transformer is None:
            pytest.skip("PyTorch not available")
        
        try:
            import torch
            
            # Test different sequence lengths
            for seq_len in [5, 20, 50]:
                test_input = torch.randn(1, seq_len, 32)  # (batch, seq, d_model)
                
                encoded = self.transformer.pos_encoding(test_input)
                
                # Should maintain shape
                assert encoded.shape == test_input.shape
                
                # Should be different from input (pos encoding added)
                assert not torch.allclose(encoded, test_input)
        
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_transformer_generation(self):
        """Test sequence generation"""
        if self.transformer is None:
            pytest.skip("PyTorch not available")
        
        try:
            import torch
            
            # Test generation from prompt
            prompt = torch.randint(0, 100, (1, 5))  # Single sequence
            
            generated = self.transformer.generate(prompt, max_new_tokens=10)
            
            # Should return longer sequence
            assert generated.shape[1] > prompt.shape[1]
            assert generated.shape[1] <= prompt.shape[1] + 10
            
            # Should start with original prompt
            assert torch.equal(generated[:, :prompt.shape[1]], prompt)
        
        except ImportError:
            pytest.skip("PyTorch not available")


class TestMetaComposer:
    """Test the meta-learning composer"""
    
    def setup_method(self):
        """Set up test composer"""
        try:
            self.composer = MetaComposer(
                vocab_size=100,
                d_model=32,
                nhead=4, 
                num_layers=2
            )
        except ImportError:
            self.composer = None
    
    def test_composer_initialization(self):
        """Test composer initializes correctly"""
        if self.composer is None:
            pytest.skip("PyTorch not available")
        
        assert self.composer is not None
        assert hasattr(self.composer, 'tokenizer')
        assert hasattr(self.composer, 'model')
        assert hasattr(self.composer, 'optimizer')
    
    def test_encode_task_pairs(self):
        """Test encoding of task input-output pairs"""
        if self.composer is None:
            pytest.skip("PyTorch not available")
        
        # Sample task pairs
        task_pairs = [
            ([[1, 0], [0, 1]], [[0, 1], [1, 0]]),  # Flip
            ([[2, 0], [0, 2]], [[0, 2], [2, 0]])   # Flip  
        ]
        
        encoded = self.composer.encode_task_pairs(task_pairs)
        
        # Should return encoded representation
        assert encoded is not None
        
        try:
            import torch
            
            if isinstance(encoded, torch.Tensor):
                assert encoded.dim() >= 1
                assert encoded.shape[0] > 0
            else:
                assert isinstance(encoded, (list, np.ndarray))
        
        except ImportError:
            pass
    
    def test_predict_primitives(self):
        """Test primitive sequence prediction"""
        if self.composer is None:
            pytest.skip("PyTorch not available")
        
        # Sample input-output pair for prediction
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[3, 1], [4, 2]]  # Rotated
        
        predicted = self.composer.predict_primitives(input_grid, output_grid)
        
        # Should return list of primitive names
        assert isinstance(predicted, list)
        
        # All predictions should be strings
        for prim in predicted:
            assert isinstance(prim, str)
        
        # Should have reasonable length
        assert len(predicted) <= 10  # Not too long
    
    def test_training_step(self):
        """Test single training step"""
        if self.composer is None:
            pytest.skip("PyTorch not available")
        
        # Create sample training data
        task_pairs = [
            ([[1, 0], [0, 1]], [[0, 1], [1, 0]]),
            ([[2, 3], [4, 5]], [[4, 2], [5, 3]])
        ]
        target_primitives = [['fh'], ['r90']]
        
        try:
            loss = self.composer.training_step(task_pairs, target_primitives)
            
            # Should return valid loss
            import torch
            assert isinstance(loss, (float, torch.Tensor))
            
            if isinstance(loss, torch.Tensor):
                assert not torch.isnan(loss)
                assert not torch.isinf(loss)
                assert loss.item() >= 0
            else:
                assert loss >= 0
                assert not np.isnan(loss)
        
        except ImportError:
            pytest.skip("PyTorch operations not available")
    
    def test_batch_training(self):
        """Test batch training process"""
        if self.composer is None:
            pytest.skip("PyTorch not available")
        
        # Create larger batch of training data
        batch_pairs = []
        batch_targets = []
        
        for i in range(5):
            input_grid = [[i, 0], [0, i]]
            output_grid = [[0, i], [i, 0]]
            batch_pairs.append((input_grid, output_grid))
            batch_targets.append(['fh'])  # All are horizontal flips
        
        try:
            # Should handle batch without errors
            loss = self.composer.training_step(batch_pairs, batch_targets)
            
            assert loss is not None
            
        except ImportError:
            pytest.skip("PyTorch operations not available") 
    
    def test_evaluation_mode(self):
        """Test switching between train/eval modes"""
        if self.composer is None:
            pytest.skip("PyTorch not available")
        
        try:
            import torch
            
            # Test eval mode
            self.composer.model.eval()
            assert not self.composer.model.training
            
            # Test train mode
            self.composer.model.train()
            assert self.composer.model.training
        
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_model_saving_loading(self):
        """Test model checkpoint save/load"""
        if self.composer is None:
            pytest.skip("PyTorch not available")
        
        try:
            import torch
            import tempfile
            
            # Save model
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                checkpoint_path = f.name
            
            self.composer.save_checkpoint(checkpoint_path)
            
            # Model file should exist
            assert Path(checkpoint_path).exists()
            
            # Load model
            self.composer.load_checkpoint(checkpoint_path)
            
            # Clean up
            Path(checkpoint_path).unlink()
            
        except ImportError:
            pytest.skip("PyTorch not available")


class TestModelIntegration:
    """Test integration between model components"""
    
    def test_tokenizer_transformer_compatibility(self):
        """Test that tokenizer output works with transformer"""
        tokenizer = ARCTokenizer()
        
        try:
            transformer = MicroTransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=32,
                nhead=4,
                num_layers=2
            )
            
            # Create sample task
            task = {
                'train': [
                    {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}
                ],
                'test': [
                    {'input': [[2, 0], [0, 2]], 'output': None}
                ]
            }
            
            # Encode task
            encoded = tokenizer.encode_arc_task(task)
            
            import torch
            
            # Convert to tensor
            input_tokens = torch.tensor([encoded['input_tokens']])
            
            # Should work with transformer
            output = transformer(input_tokens)
            
            assert output is not None
            assert output.shape[0] == 1  # Batch size
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_end_to_end_pipeline(self):
        """Test complete model pipeline"""
        try:
            composer = MetaComposer(
                vocab_size=100,
                d_model=32,
                nhead=4,
                num_layers=2
            )
            
            # Sample training data
            input_grid = [[1, 2], [3, 4]]
            output_grid = [[3, 1], [4, 2]]
            
            # Should run prediction without errors
            predictions = composer.predict_primitives(input_grid, output_grid)
            
            assert isinstance(predictions, list)
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_memory_efficiency(self):
        """Test that models are memory efficient"""
        try:
            import torch
            
            # Create model with constraints
            composer = MetaComposer(
                vocab_size=50,  # Smaller vocab
                d_model=16,     # Smaller model
                nhead=2,        # Fewer heads
                num_layers=1    # Single layer
            )
            
            # Check parameter count
            total_params = sum(p.numel() for p in composer.model.parameters())
            assert total_params < 10000  # Very small model
            
            # Test memory usage with batch
            batch_size = 10
            seq_len = 20
            test_input = torch.randint(0, 50, (batch_size, seq_len))
            
            # Should run without memory issues
            output = composer.model(test_input)
            assert output.shape == (batch_size, seq_len, 50)
            
        except ImportError:
            pytest.skip("PyTorch not available")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
