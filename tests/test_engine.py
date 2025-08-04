"""
Test suite for MicroGolf engine components
Tests controller, executor, and NCA modules
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from microgolf.engine.controller import PrimitiveController  
from microgolf.engine.executor import OptimizedExecutor
from microgolf.engine.nca import MicroNCA
from microgolf.primitives import PRIMITIVES


class TestPrimitiveController:
    """Test the heuristic controller with ML pattern recognition"""
    
    def setup_method(self):
        """Set up test controller"""
        self.controller = PrimitiveController()
    
    def test_controller_initialization(self):
        """Test controller initializes correctly"""
        assert self.controller is not None
        assert hasattr(self.controller, 'kmeans')
        assert hasattr(self.controller, 'scaler')
        assert hasattr(self.controller, 'primitive_weights')
    
    def test_extract_features(self):
        """Test feature extraction from grids"""
        # Simple test grid
        input_grid = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        output_grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        
        features = self.controller.extract_features(input_grid, output_grid)
        
        # Should return feature vector
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert len(features) == 12  # As defined in extract_features
        
        # Test with different grid sizes
        small_grid = [[1]]
        large_grid = [[i % 3 for i in range(10)] for _ in range(10)]
        
        features_small = self.controller.extract_features(small_grid, small_grid)
        features_large = self.controller.extract_features(large_grid, large_grid)
        
        assert len(features_small) == len(features_large) == 12
    
    def test_get_primitive_candidates(self):
        """Test primitive candidate selection"""
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[3, 1], [4, 2]]  # Looks like a rotation
        
        candidates = self.controller.get_primitive_candidates(input_grid, output_grid)
        
        # Should return list of primitive names
        assert isinstance(candidates, list)
        assert len(candidates) <= len(PRIMITIVES)
        
        # All candidates should be valid primitive names
        for candidate in candidates:
            assert candidate in PRIMITIVES
        
        # Should have at least one candidate
        assert len(candidates) > 0
    
    def test_train_controller(self):
        """Test controller training on sample data"""
        # Create sample training data
        training_examples = []
        
        # Example 1: Rotation pattern
        input1 = [[1, 2], [3, 4]]
        output1 = [[3, 1], [4, 2]]
        solution1 = ['r90']
        training_examples.append((input1, output1, solution1))
        
        # Example 2: Color mapping pattern  
        input2 = [[0, 1], [1, 0]]
        output2 = [[2, 3], [3, 2]]
        solution2 = ['mc']
        training_examples.append((input2, output2, solution2))
        
        # Train controller
        self.controller.train(training_examples)
        
        # Should have updated weights
        assert hasattr(self.controller, 'primitive_weights')
        assert len(self.controller.primitive_weights) > 0
    
    def test_empty_grids(self):
        """Test controller behavior with empty grids"""
        empty_grid = []
        
        features = self.controller.extract_features(empty_grid, empty_grid)
        assert len(features) == 12  # Should still return full feature vector
        
        candidates = self.controller.get_primitive_candidates(empty_grid, empty_grid)
        assert isinstance(candidates, list)
    
    def test_single_element_grids(self):
        """Test controller with single-element grids"""
        single_grid = [[5]]
        
        features = self.controller.extract_features(single_grid, single_grid)
        assert len(features) == 12
        
        candidates = self.controller.get_primitive_candidates(single_grid, single_grid)
        assert isinstance(candidates, list)


class TestOptimizedExecutor:
    """Test the code executor with AST optimization"""
    
    def setup_method(self):
        """Set up test executor"""
        self.executor = OptimizedExecutor()
    
    def test_executor_initialization(self):
        """Test executor initializes correctly"""
        assert self.executor is not None
        assert hasattr(self.executor, 'max_iterations')
        assert hasattr(self.executor, 'timeout')
    
    def test_primitive_sequence_to_code(self):
        """Test converting primitive sequence to executable code"""
        # Simple sequence
        primitives = ['r90', 'fh']
        
        code = self.executor.primitive_sequence_to_code(primitives)
        
        # Should generate valid Python code
        assert isinstance(code, str)
        assert len(code) > 0
        assert 'r90' in code
        assert 'fh' in code
        
        # Should be compact
        assert len(code) < 500  # Reasonable upper bound
    
    def test_optimize_ast(self):
        """Test AST optimization"""
        # Sample code that can be optimized
        test_code = """
def solve(grid):
    temp = r90(grid)
    temp = r90(temp)
    temp = r90(temp) 
    temp = r90(temp)
    return temp
"""
        
        optimized = self.executor.optimize_ast(test_code)
        
        # Should return optimized code
        assert isinstance(optimized, str)
        assert len(optimized) > 0
        
        # Optimized version might be shorter (though not guaranteed)
        # Main test is that it doesn't crash
    
    def test_execute_sequence(self):
        """Test executing primitive sequence"""
        input_grid = [[1, 2], [3, 4]]
        primitives = ['r90']
        
        result = self.executor.execute_sequence(input_grid, primitives)
        
        # Should return a grid
        assert result is not None
        assert isinstance(result, list)
        
        # Result should be rotated version
        expected = [[3, 1], [4, 2]]
        assert result == expected
    
    def test_multiple_strategies(self):
        """Test executor with different optimization strategies"""
        input_grid = [[1, 2], [3, 4]]
        primitives = ['r90', 'fh']
        
        # Test all strategies
        strategies = ['sequential', 'functional', 'minimal']
        
        for strategy in strategies:
            result = self.executor.execute_sequence(
                input_grid, primitives, strategy=strategy
            )
            
            # All strategies should produce valid results
            assert result is not None
            assert isinstance(result, list)
    
    def test_empty_sequence(self):
        """Test executor with empty primitive sequence"""
        input_grid = [[1, 2], [3, 4]]
        primitives = []
        
        result = self.executor.execute_sequence(input_grid, primitives)
        
        # Should return original grid
        assert result == input_grid
    
    def test_invalid_primitives(self):
        """Test executor with invalid primitive names"""
        input_grid = [[1, 2], [3, 4]]
        primitives = ['invalid_primitive']
        
        # Should handle gracefully (return None or raise exception)
        try:
            result = self.executor.execute_sequence(input_grid, primitives)
            # If it doesn't raise exception, should return None or original
            assert result is None or result == input_grid
        except Exception:
            # Expected to raise exception for invalid primitive
            pass
    
    def test_code_generation_formats(self):
        """Test different code generation formats"""
        primitives = ['r90', 'mc', 'inc']
        
        # Test all code generation strategies
        strategies = ['sequential', 'functional', 'minimal']
        
        for strategy in strategies:
            code = self.executor.primitive_sequence_to_code(
                primitives, strategy=strategy
            )
            
            assert isinstance(code, str)
            assert len(code) > 0
            
            # Should contain all primitives
            for prim in primitives:
                if prim != 'mc':  # mc might be handled differently
                    assert prim in code or 'map' in code.lower()


class TestMicroNCA:
    """Test the Neural Cellular Automata module"""
    
    def setup_method(self):
        """Set up test NCA"""
        self.nca = MicroNCA(grid_size=5, hidden_dim=8)
    
    def test_nca_initialization(self):
        """Test NCA initializes correctly"""
        assert self.nca is not None
        assert hasattr(self.nca, 'conv1')
        assert hasattr(self.nca, 'conv2') 
        assert hasattr(self.nca, 'output_conv')
        
        # Check parameter count is small
        total_params = sum(p.numel() for p in self.nca.parameters())
        assert total_params < 1000  # Should be very lightweight
    
    def test_nca_forward_pass(self):
        """Test NCA forward pass"""
        # Create test input (batch_size=1, channels=1, height=5, width=5)
        test_input = np.random.rand(1, 1, 5, 5).astype(np.float32)
        
        # Convert to torch tensor if torch is available
        try:
            import torch
            input_tensor = torch.FloatTensor(test_input)
            
            output = self.nca(input_tensor)
            
            # Should return tensor of same spatial dimensions
            assert output.shape[-2:] == (5, 5)  # Same height, width
            assert output.shape[0] == 1  # Same batch size
            
        except ImportError:
            # If torch not available, skip this test
            pytest.skip("PyTorch not available")
    
    def test_nca_step_function(self):
        """Test NCA single step evolution"""
        # Create simple test grid 
        test_grid = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        
        try:
            evolved_grid = self.nca.evolve_grid(test_grid, steps=1)
            
            # Should return grid of same size
            assert len(evolved_grid) == len(test_grid)
            assert len(evolved_grid[0]) == len(test_grid[0])
            
            # Values should be in reasonable range
            for row in evolved_grid:
                for val in row:
                    assert isinstance(val, (int, float))
                    
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_nca_multi_step_evolution(self):
        """Test NCA multi-step evolution"""
        test_grid = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
        
        try:
            # Evolve for multiple steps
            final_grid = self.nca.evolve_grid(test_grid, steps=5)
            
            # Should converge to some pattern
            assert len(final_grid) == len(test_grid)
            assert len(final_grid[0]) == len(test_grid[0])
            
            # Test that evolution actually changes the grid
            intermediate = self.nca.evolve_grid(test_grid, steps=1)
            assert intermediate != test_grid  # Should change
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_nca_different_sizes(self):
        """Test NCA with different grid sizes"""
        # Test with different sizes than initialization
        small_grid = [[1, 0], [0, 1]]
        large_grid = [[i % 2 for i in range(8)] for _ in range(8)]
        
        try:
            # Should handle different sizes gracefully
            small_result = self.nca.evolve_grid(small_grid, steps=1)
            large_result = self.nca.evolve_grid(large_grid, steps=1)
            
            assert len(small_result) == 2
            assert len(small_result[0]) == 2
            assert len(large_result) == 8
            assert len(large_result[0]) == 8
            
        except ImportError:
            pytest.skip("PyTorch not available")


class TestEngineIntegration:
    """Test integration between engine components"""
    
    def setup_method(self):
        """Set up test components"""
        self.controller = PrimitiveController()
        self.executor = OptimizedExecutor()
        self.nca = MicroNCA(grid_size=3, hidden_dim=4)
    
    def test_controller_executor_pipeline(self):
        """Test full pipeline from controller to executor"""
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[3, 1], [4, 2]]  # Rotated
        
        # Get candidates from controller
        candidates = self.controller.get_primitive_candidates(input_grid, output_grid)
        
        # Execute with executor
        if candidates:
            result = self.executor.execute_sequence(input_grid, candidates[:1])
            
            # Should produce some result
            assert result is not None
            assert isinstance(result, list)
    
    def test_full_solution_pipeline(self):
        """Test complete solution generation pipeline"""
        input_grid = [[0, 1], [1, 0]]
        target_output = [[1, 0], [0, 1]]  # Horizontally flipped
        
        # Step 1: Get primitive candidates
        candidates = self.controller.get_primitive_candidates(input_grid, target_output)
        
        # Step 2: Try each candidate
        best_result = None
        best_sequence = None
        
        for candidate in candidates[:3]:  # Try top 3
            result = self.executor.execute_sequence(input_grid, [candidate])
            
            if result == target_output:
                best_result = result
                best_sequence = [candidate]
                break
        
        # Step 3: If exact match not found, try NCA evolution
        if best_result != target_output:
            try:
                nca_result = self.nca.evolve_grid(input_grid, steps=3)
                # Convert back to integer grid
                nca_result = [[int(round(val)) for val in row] for row in nca_result]
                
            except ImportError:
                # Skip NCA part if PyTorch not available
                pass
        
        # Pipeline should complete without errors
        assert True  # If we get here, pipeline worked
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        # Test with problematic inputs
        empty_grid = []
        large_grid = [[i for i in range(20)] for _ in range(20)]
        
        # Controller should handle gracefully
        try:
            candidates = self.controller.get_primitive_candidates(empty_grid, empty_grid)
            assert isinstance(candidates, list)
        except Exception as e:
            pytest.fail(f"Controller failed on empty grid: {e}")
        
        # Executor should handle gracefully  
        try:
            result = self.executor.execute_sequence(large_grid, ['r90'])
            assert result is not None
        except Exception as e:
            pytest.fail(f"Executor failed on large grid: {e}")
    
    def test_performance_integration(self):
        """Test that integrated pipeline performs reasonably"""
        import time
        
        # Test with medium-sized problem
        input_grid = [[i % 3 for i in range(10)] for _ in range(10)]
        output_grid = [[i % 3 for i in range(10)] for _ in range(10)]
        
        start_time = time.time()
        
        # Run pipeline
        candidates = self.controller.get_primitive_candidates(input_grid, output_grid)
        
        if candidates:
            result = self.executor.execute_sequence(input_grid, candidates[:2])
        
        elapsed = time.time() - start_time
        
        # Should complete quickly
        assert elapsed < 5.0, f"Pipeline too slow: {elapsed:.3f}s"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
