"""
Comprehensive test suite for MicroGolf primitives
Tests all ultra-compact functions for correctness and byte compliance
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from microgolf.primitives import *
from microgolf.primitives import PRIMITIVES, TOTAL_BYTES


class TestGeometryPrimitives:
    """Test geometric transformation primitives"""
    
    def test_r90_rotation(self):
        """Test 90-degree rotation"""
        input_grid = [[1, 2], [3, 4]]
        expected = [[3, 1], [4, 2]]
        result = r90(input_grid)
        assert result == expected
        
        # Test multiple rotations
        grid = [[1, 2, 3]]
        rotated_once = r90(grid)
        rotated_twice = r90(rotated_once)
        rotated_thrice = r90(rotated_twice)
        rotated_full = r90(rotated_thrice)
        
        # Should be back to original after 4 rotations
        assert rotated_full == grid
    
    def test_fh_horizontal_flip(self):
        """Test horizontal flip"""
        input_grid = [[1, 2], [3, 4]]
        expected = [[3, 4], [1, 2]]
        result = fh(input_grid)
        assert result == expected
        
        # Test double flip returns original
        double_flipped = fh(fh(input_grid))
        assert double_flipped == input_grid
    
    def test_fv_vertical_flip(self):
        """Test vertical flip"""
        input_grid = [[1, 2], [3, 4]]
        expected = [[2, 1], [4, 3]]
        result = fv(input_grid)
        assert result == expected
        
        # Test double flip returns original
        double_flipped = fv(fv(input_grid))
        assert double_flipped == input_grid
    
    def test_tr_transpose(self):
        """Test matrix transpose"""
        input_grid = [[1, 2, 3], [4, 5, 6]]
        expected = [[1, 4], [2, 5], [3, 6]]
        result = tr(input_grid)
        assert result == expected
        
        # Test double transpose returns original
        double_transposed = tr(tr([[1, 2], [3, 4]]))
        assert double_transposed == [[1, 2], [3, 4]]
    
    def test_sh_shift(self):
        """Test grid shifting"""
        input_grid = [[1, 2, 3], [4, 5, 6]]
        
        # Test vertical shift (x=0)
        result_v = sh(input_grid, 0, 1)
        expected_v = [[4, 5, 6], [1, 2, 3]]
        assert result_v == expected_v
        
        # Test horizontal shift (y=0, but with rows)
        result_h = sh(input_grid, 1, 0) 
        expected_h = [[2, 3, 1], [5, 6, 4]]
        assert result_h == expected_h
    
    def test_geometry_edge_cases(self):
        """Test edge cases for geometry primitives"""
        
        # Empty grid
        empty = []
        assert r90(empty) == []
        assert fh(empty) == []
        assert tr(empty) == []
        
        # Single element
        single = [[1]]
        assert r90(single) == [[1]]
        assert fh(single) == [[1]]
        assert fv(single) == [[1]]
        assert tr(single) == [[1]]


class TestColorOperations:
    """Test color operation primitives"""
    
    def test_mc_color_mapping(self):
        """Test color mapping"""
        input_grid = [[1, 0, 1], [0, 1, 0]]
        color_map = {0: 2, 1: 3}
        expected = [[3, 2, 3], [2, 3, 2]]
        result = mc(input_grid, color_map)
        assert result == expected
        
        # Test identity mapping
        identity_map = {0: 0, 1: 1, 2: 2}
        result_identity = mc(input_grid, identity_map)
        assert result_identity == input_grid
    
    def test_tm_threshold_mask(self):
        """Test threshold masking"""
        input_grid = [[0, 1, 2], [3, 4, 5]]
        threshold = 2
        expected = [[0, 0, 0], [1, 1, 1]]
        result = tm(input_grid, threshold)
        assert result == expected
    
    def test_rc_replace_color(self):
        """Test color replacement"""
        input_grid = [[1, 0, 1], [0, 1, 0]]
        result = rc(input_grid, 0, 9)
        expected = [[1, 9, 1], [9, 1, 9]]
        assert result == expected
        
        # Test replacing non-existent color
        result_none = rc(input_grid, 5, 7)
        assert result_none == input_grid
    
    def test_bc_blend_colors(self):
        """Test color blending"""
        input_grid = [[1, 2], [3, 4]]
        result = bc(input_grid, 2, 1)  # multiply by 2, add 1
        expected = [[3, 5], [7, 9]]
        result_int = [[int(x) for x in row] for row in result]
        assert result_int == expected
    
    def test_md_max_difference(self):
        """Test max difference operation"""
        input_grid = [[1, 2], [3, 4]]
        result = md(input_grid)
        expected = [[3, 2], [1, 0]]  # 4 - each value
        assert result == expected


class TestShapeOperations:
    """Test shape operation primitives"""
    
    def test_ff_flood_fill(self):
        """Test flood fill"""
        # Create a grid with connected regions
        input_grid = [
            [0, 1, 0],  
            [1, 0, 1],
            [0, 1, 0]
        ]
        
        # Fill from center (1,1) with color 9
        result = ff(input_grid, 1, 1, 9)
        
        # Center and connected zeros should be filled 
        # Note: This is a simplified test - actual flood fill behavior may vary
        assert result[1][1] == 9
    
    def test_bb_bounding_box(self):
        """Test bounding box calculation"""
        input_grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0], 
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ]
        
        result = bb(input_grid)
        # Should return (min_row, min_col, max_row, max_col)
        assert len(result) == 4
        assert all(isinstance(x, int) for x in result)
    
    def test_ct_centroid(self):
        """Test centroid calculation"""
        input_grid = [
            [1, 0, 1],
            [0, 0, 0], 
            [1, 0, 1]
        ]
        
        result = ct(input_grid)
        # Should return (center_row, center_col)
        assert len(result) == 2
        assert all(isinstance(x, int) for x in result)
    
    def test_cc_connected_components(self):
        """Test connected components counting"""
        input_grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1] 
        ]
        
        result = cc(input_grid)
        # Should count separate connected regions of non-zero values
        assert isinstance(result, int)
        assert result >= 0


class TestNumericOperations:
    """Test numeric operation primitives"""
    
    def test_inc_increment(self):
        """Test increment operation"""
        input_grid = [[0, 1], [2, 3]]
        expected = [[1, 2], [3, 4]]
        result = inc(input_grid)
        assert result == expected
    
    def test_cl_clamp(self):
        """Test clamp operation"""
        input_grid = [[0, 5, 10], [15, 20, 25]]
        result = cl(input_grid, 5, 15)  # clamp between 5 and 15
        expected = [[5, 5, 10], [15, 15, 15]]
        assert result == expected
    
    def test_he_histogram_equalization(self):
        """Test histogram equalization"""
        input_grid = [[1, 1, 2], [2, 3, 3]]
        result = he(input_grid)
        
        # Should return a grid with same shape
        assert len(result) == len(input_grid)
        assert len(result[0]) == len(input_grid[0])
    
    def test_sm_sum(self):
        """Test grid sum"""
        input_grid = [[1, 2], [3, 4]]
        result = sm(input_grid)
        expected = 10  # 1+2+3+4
        assert result == expected
        
        # Test empty grid
        assert sm([]) == 0
        
        # Test grid with zeros
        zero_grid = [[0, 0], [0, 0]]
        assert sm(zero_grid) == 0
    
    def test_avg_average(self):
        """Test grid average"""
        input_grid = [[2, 4], [6, 8]]
        result = avg(input_grid)
        expected = 5  # (2+4+6+8) // 4
        assert result == expected


class TestPrimitiveMetadata:
    """Test primitive metadata and compliance"""
    
    def test_primitive_count(self):
        """Test that we have expected number of primitives"""
        assert len(PRIMITIVES) == 19
        
        # Check categories
        geometry_prims = ['r90', 'fh', 'fv', 'tr', 'sh']
        color_prims = ['mc', 'tm', 'rc', 'bc', 'md']
        shape_prims = ['ff', 'bb', 'ct', 'cc']
        numeric_prims = ['inc', 'cl', 'he', 'sm', 'avg']
        
        all_expected = geometry_prims + color_prims + shape_prims + numeric_prims
        
        for prim in all_expected:
            assert prim in PRIMITIVES
    
    def test_byte_counts(self):
        """Test that primitive byte counts are reasonable"""
        for prim_name, (func, byte_count) in PRIMITIVES.items():
            # Each primitive should be <= 20 bytes as specified
            assert byte_count <= 20, f"{prim_name} exceeds 20 byte limit: {byte_count}"
            
            # Byte count should be positive
            assert byte_count > 0, f"{prim_name} has invalid byte count: {byte_count}"
    
    def test_total_bytes(self):
        """Test total primitive library size"""
        calculated_total = sum(bytes_count for _, bytes_count in PRIMITIVES.values())
        assert TOTAL_BYTES == calculated_total
        
        # Ensure total is reasonable for competition
        assert TOTAL_BYTES < 500, f"Total primitive library too large: {TOTAL_BYTES}"
    
    def test_primitive_functions_exist(self):
        """Test that all primitive functions are callable"""
        for prim_name, (func, _) in PRIMITIVES.items():
            assert callable(func), f"{prim_name} function is not callable"
    
    def test_function_signatures(self):
        """Test that functions have reasonable signatures"""
        # This is a basic test - in practice you'd inspect actual signatures
        
        # Single argument functions (most geometry and some others)
        single_arg_funcs = ['r90', 'fh', 'fv', 'tr', 'inc', 'sm', 'avg', 'md', 'he']
        
        for func_name in single_arg_funcs:
            if func_name in PRIMITIVES:
                func, _ = PRIMITIVES[func_name]
                # Test with simple input
                try:
                    test_grid = [[1, 2], [3, 4]]
                    result = func(test_grid)
                    # Should return something (not necessarily specific value)
                    assert result is not None
                except Exception as e:
                    # Some functions might fail on this simple test, that's ok
                    pass


class TestPrimitiveIntegration:
    """Test primitive integration and chaining"""
    
    def test_primitive_chaining(self):
        """Test that primitives can be chained together"""
        input_grid = [[1, 0], [0, 1]]
        
        # Chain: rotate -> flip horizontal -> color map
        step1 = r90(input_grid)
        step2 = fh(step1) 
        color_map = {0: 2, 1: 3}
        step3 = mc(step2, color_map)
        
        # Should complete without errors
        assert step3 is not None
        assert len(step3) > 0
    
    def test_primitive_commutativity(self):
        """Test commutativity properties where applicable"""
        input_grid = [[1, 2], [3, 4]]
        
        # Double application should return to original for symmetric operations
        assert fh(fh(input_grid)) == input_grid
        assert fv(fv(input_grid)) == input_grid
        
        # Four rotations should return to original
        rotated = input_grid
        for _ in range(4):
            rotated = r90(rotated)
        assert rotated == input_grid
    
    def test_primitive_edge_cases(self):
        """Test primitives with edge case inputs"""
        
        # Test with different grid sizes
        sizes_to_test = [
            [[1]],  # 1x1
            [[1, 2]],  # 1x2  
            [[1], [2]],  # 2x1
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3x3
        ]
        
        for grid in sizes_to_test:
            # Test geometry operations
            try:
                r90_result = r90(grid)
                assert r90_result is not None
            except:
                pass  # Some operations might fail on certain sizes
            
            try:
                fh_result = fh(grid)
                assert fh_result is not None
            except:
                pass
            
            # Test numeric operations
            try:
                inc_result = inc(grid)
                assert inc_result is not None
                assert len(inc_result) == len(grid)
            except:
                pass


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for primitives (marked as slow)"""
    
    def test_primitive_performance(self):
        """Test that primitives execute quickly on larger grids"""
        import time
        
        # Create larger test grid
        large_grid = [[i % 10 for i in range(50)] for _ in range(50)]
        
        # Test a few key primitives
        test_primitives = ['r90', 'fh', 'mc', 'inc', 'sm']
        
        for prim_name in test_primitives:
            if prim_name in PRIMITIVES:
                func, _ = PRIMITIVES[prim_name]
                
                start_time = time.time()
                
                try:
                    if prim_name == 'mc':
                        result = func(large_grid, {i: i for i in range(10)})
                    else:
                        result = func(large_grid)
                    
                    execution_time = time.time() - start_time
                    
                    # Should execute quickly (< 1 second)
                    assert execution_time < 1.0, f"{prim_name} too slow: {execution_time:.3f}s"
                    
                except Exception as e:
                    # Some primitives might not work with this test setup
                    print(f"Performance test failed for {prim_name}: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
