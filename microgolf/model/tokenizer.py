"""
Advanced Tokenizer for ARC-AGI Tasks
Converts grid examples into compact token sequences for meta-learning
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json


class ARCTokenizer:
    """Converts ARC grids into compact token sequences using multiple encoding strategies"""
    
    def __init__(self, max_tokens: int = 100):
        self.max_tokens = max_tokens
        self.vocab_size = 256  # Sufficient for complex patterns
        
        # Special tokens
        self.SPECIAL_TOKENS = {
            'PAD': 0,
            'SEP': 1,    # Separates input/output
            'BOG': 2,    # Begin grid
            'EOG': 3,    # End grid
            'RLE': 4,    # Run-length encoding marker
            'CLUSTER': 5,  # Color cluster marker
            'REPEAT': 6,   # Pattern repeat marker
        }
        
        # Color tokens (10-19)
        self.COLOR_OFFSET = 10
        
        # Position encoding tokens (20-99)
        self.POS_OFFSET = 20
        
        # Pattern tokens (100-199)
        self.PATTERN_OFFSET = 100
        
        # Transformation tokens (200-255)
        self.TRANSFORM_OFFSET = 200
        
    def tokenize_examples(self, examples: List[Dict]) -> List[int]:
        """Tokenize multiple input/output examples into unified sequence"""
        tokens = []
        
        for i, example in enumerate(examples[:3]):  # Max 3 examples
            inp, out = example['input'], example['output']
            
            # Add example separator for multi-example sequences
            if i > 0:
                tokens.append(self.SPECIAL_TOKENS['SEP'])
            
            # Tokenize input grid
            tokens.extend(self._tokenize_grid(inp, is_input=True))
            
            # Add input/output separator
            tokens.append(self.SPECIAL_TOKENS['SEP'])
            
            # Tokenize output grid
            tokens.extend(self._tokenize_grid(out, is_input=False))
        
        # Truncate or pad to max_tokens
        return self._normalize_length(tokens)
    
    def _tokenize_grid(self, grid: List[List[int]], is_input: bool = True) -> List[int]:
        """Tokenize single grid using multiple encoding strategies"""
        if not grid or not grid[0]:
            return [self.SPECIAL_TOKENS['BOG'], self.SPECIAL_TOKENS['EOG']]
        
        tokens = [self.SPECIAL_TOKENS['BOG']]
        
        # Choose best encoding strategy based on grid properties
        strategy = self._select_encoding_strategy(grid)
        
        if strategy == 'rle':
            tokens.extend(self._rle_encode(grid))
        elif strategy == 'cluster':
            tokens.extend(self._cluster_encode(grid))
        elif strategy == 'pattern':
            tokens.extend(self._pattern_encode(grid))
        else:  # fallback to direct encoding
            tokens.extend(self._direct_encode(grid))
        
        tokens.append(self.SPECIAL_TOKENS['EOG'])
        return tokens
    
    def _select_encoding_strategy(self, grid: List[List[int]]) -> str:
        """Select optimal encoding strategy based on grid characteristics"""
        h, w = len(grid), len(grid[0])
        total_cells = h * w
        
        # Analyze grid properties
        colors = set(cell for row in grid for cell in row)
        color_count = len(colors)
        
        # Count runs for RLE efficiency
        run_count = self._count_runs(grid)
        rle_efficiency = run_count / total_cells
        
        # Count connected components for cluster efficiency
        component_count = self._count_components(grid)
        cluster_efficiency = component_count / total_cells
        
        # Detect patterns
        has_patterns = self._detect_patterns(grid)
        
        # Selection logic
        if has_patterns and total_cells > 16:
            return 'pattern'
        elif rle_efficiency < 0.7 and run_count < total_cells * 0.8:
            return 'rle'
        elif cluster_efficiency < 0.5 and component_count < 10:
            return 'cluster'
        else:
            return 'direct'
    
    def _rle_encode(self, grid: List[List[int]]) -> List[int]:
        """Run-length encoding for grids with repeated values"""
        tokens = [self.SPECIAL_TOKENS['RLE']]
        
        # Flatten grid and apply RLE
        flat = [cell for row in grid for cell in row]
        
        i = 0
        while i < len(flat) and len(tokens) < self.max_tokens - 10:
            color = flat[i]
            count = 1
            
            # Count consecutive cells of same color
            while i + count < len(flat) and flat[i + count] == color and count < 63:
                count += 1
            
            # Encode as color + count
            tokens.append(self.COLOR_OFFSET + color)
            tokens.append(count)
            
            i += count
        
        return tokens
    
    def _cluster_encode(self, grid: List[List[int]]) -> List[int]:
        """Encode based on connected color clusters"""
        tokens = [self.SPECIAL_TOKENS['CLUSTER']]
        
        h, w = len(grid), len(grid[0])
        visited = set()
        
        for i in range(h):
            for j in range(w):
                if (i, j) not in visited and grid[i][j] != 0:
                    # Find connected component
                    component = self._flood_fill_component(grid, i, j, visited)
                    
                    if len(tokens) >= self.max_tokens - 10:
                        break
                    
                    # Encode cluster: color, size, bounding box
                    color = grid[i][j]
                    size = len(component)
                    
                    min_i = min(pos[0] for pos in component)
                    min_j = min(pos[1] for pos in component)
                    max_i = max(pos[0] for pos in component)
                    max_j = max(pos[1] for pos in component)
                    
                    tokens.extend([
                        self.COLOR_OFFSET + color,
                        min(size, 63),  # Clamp size
                        self.POS_OFFSET + min_i,
                        self.POS_OFFSET + min_j,
                        self.POS_OFFSET + (max_i - min_i),
                        self.POS_OFFSET + (max_j - min_j)
                    ])
        
        return tokens
    
    def _pattern_encode(self, grid: List[List[int]]) -> List[int]:
        """Encode based on detected patterns and symmetries"""
        tokens = [self.SPECIAL_TOKENS['REPEAT']]
        
        # Detect common patterns
        patterns = self._detect_grid_patterns(grid)
        
        for pattern_type, params in patterns:
            if len(tokens) >= self.max_tokens - 5:
                break
            
            tokens.append(self.PATTERN_OFFSET + pattern_type)
            tokens.extend([min(p, 255) for p in params[:3]])  # Max 3 params
        
        return tokens
    
    def _direct_encode(self, grid: List[List[int]]) -> List[int]:
        """Direct encoding with position information"""
        tokens = []
        h, w = len(grid), len(grid[0])
        
        # Encode dimensions first
        tokens.extend([self.POS_OFFSET + h, self.POS_OFFSET + w])
        
        # Encode non-zero cells with positions
        for i in range(h):
            for j in range(w):
                if grid[i][j] != 0 and len(tokens) < self.max_tokens - 5:
                    tokens.extend([
                        self.COLOR_OFFSET + grid[i][j],
                        self.POS_OFFSET + i,
                        self.POS_OFFSET + j
                    ])
        
        return tokens
    
    def _count_runs(self, grid: List[List[int]]) -> int:
        """Count number of runs for RLE efficiency estimation"""
        runs = 0
        flat = [cell for row in grid for cell in row]
        
        if not flat:
            return 0
        
        current = flat[0]
        runs = 1
        
        for cell in flat[1:]:
            if cell != current:
                runs += 1
                current = cell
        
        return runs
    
    def _count_components(self, grid: List[List[int]]) -> int:
        """Count connected components"""
        if not grid or not grid[0]:
            return 0
        
        h, w = len(grid), len(grid[0])
        visited = set()
        components = 0
        
        for i in range(h):
            for j in range(w):
                if (i, j) not in visited and grid[i][j] != 0:
                    self._flood_fill_component(grid, i, j, visited)
                    components += 1
        
        return components
    
    def _flood_fill_component(self, grid: List[List[int]], start_i: int, start_j: int, 
                            visited: set) -> List[Tuple[int, int]]:
        """Find connected component using flood fill"""
        h, w = len(grid), len(grid[0])
        color = grid[start_i][start_j]
        component = []
        
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i, j) in visited or i < 0 or i >= h or j < 0 or j >= w:
                continue
            if grid[i][j] != color:
                continue
            
            visited.add((i, j))
            component.append((i, j))
            
            # Add neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((i + di, j + dj))
        
        return component
    
    def _detect_patterns(self, grid: List[List[int]]) -> bool:
        """Detect if grid contains repeating patterns"""
        h, w = len(grid), len(grid[0])
        
        # Check for horizontal patterns
        if w >= 4:
            for pattern_len in [2, 3]:
                if w % pattern_len == 0:
                    is_pattern = True
                    for i in range(h):
                        for j in range(w):
                            if grid[i][j] != grid[i][j % pattern_len]:
                                is_pattern = False
                                break
                        if not is_pattern:
                            break
                    if is_pattern:
                        return True
        
        # Check for vertical patterns
        if h >= 4:
            for pattern_len in [2, 3]:
                if h % pattern_len == 0:
                    is_pattern = True
                    for i in range(h):
                        for j in range(w):
                            if grid[i][j] != grid[i % pattern_len][j]:
                                is_pattern = False
                                break
                        if not is_pattern:
                            break
                    if is_pattern:
                        return True
        
        return False
    
    def _detect_grid_patterns(self, grid: List[List[int]]) -> List[Tuple[int, List[int]]]:
        """Detect specific patterns in grid"""
        patterns = []
        h, w = len(grid), len(grid[0])
        
        # Pattern types (encoded as indices)
        HORIZONTAL_REPEAT = 1
        VERTICAL_REPEAT = 2
        DIAGONAL_SYMMETRY = 3
        ROTATIONAL_SYMMETRY = 4
        
        # Check horizontal repetition
        for period in [2, 3, 4]:
            if w % period == 0:
                is_repeat = all(
                    grid[i][j] == grid[i][j % period]
                    for i in range(h) for j in range(w)
                )
                if is_repeat:
                    patterns.append((HORIZONTAL_REPEAT, [period, h, w]))
                    break
        
        # Check vertical repetition
        for period in [2, 3, 4]:
            if h % period == 0:
                is_repeat = all(
                    grid[i][j] == grid[i % period][j]
                    for i in range(h) for j in range(w)
                )
                if is_repeat:
                    patterns.append((VERTICAL_REPEAT, [period, h, w]))
                    break
        
        # Check symmetries (if square grid)
        if h == w:
            # Diagonal symmetry
            is_diag_sym = all(
                grid[i][j] == grid[j][i]
                for i in range(h) for j in range(w)
            )
            if is_diag_sym:
                patterns.append((DIAGONAL_SYMMETRY, [h, w, 0]))
            
            # 180-degree rotational symmetry
            is_rot_sym = all(
                grid[i][j] == grid[h-1-i][w-1-j]
                for i in range(h) for j in range(w)
            )
            if is_rot_sym:
                patterns.append((ROTATIONAL_SYMMETRY, [h, w, 180]))
        
        return patterns
    
    def _normalize_length(self, tokens: List[int]) -> List[int]:
        """Normalize token sequence to fixed length"""
        if len(tokens) > self.max_tokens:
            return tokens[:self.max_tokens]
        else:
            padding = [self.SPECIAL_TOKENS['PAD']] * (self.max_tokens - len(tokens))
            return tokens + padding
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to readable format (for debugging)"""
        decoded = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token in self.SPECIAL_TOKENS.values():
                # Special token
                name = next(k for k, v in self.SPECIAL_TOKENS.items() if v == token)
                decoded.append(f"<{name}>")
                i += 1
            elif self.COLOR_OFFSET <= token < self.COLOR_OFFSET + 10:
                # Color token
                color = token - self.COLOR_OFFSET
                decoded.append(f"C{color}")
                i += 1
            elif self.POS_OFFSET <= token < self.POS_OFFSET + 80:
                # Position token
                pos = token - self.POS_OFFSET
                decoded.append(f"P{pos}")
                i += 1
            elif self.PATTERN_OFFSET <= token < self.PATTERN_OFFSET + 100:
                # Pattern token
                pattern = token - self.PATTERN_OFFSET
                decoded.append(f"PAT{pattern}")
                i += 1
            else:
                decoded.append(f"T{token}")
                i += 1
        
        return " ".join(decoded)


class FeatureExtractor:
    """Extract additional features for meta-learning beyond tokenization"""
    
    def __init__(self):
        self.feature_dim = 64
    
    def extract_features(self, examples: List[Dict]) -> np.ndarray:
        """Extract comprehensive feature vector from examples"""
        features = []
        
        for example in examples[:3]:  # Max 3 examples
            inp, out = example['input'], example['output']
            
            # Basic statistics (8 features)
            ih, iw = len(inp), len(inp[0]) if inp else 0
            oh, ow = len(out), len(out[0]) if out else 0
            
            basic_stats = [
                ih, iw, oh, ow,
                ih * iw, oh * ow,  # areas
                abs(oh - ih), abs(ow - iw)  # size changes
            ]
            
            # Color statistics (8 features)
            inp_colors = set(c for row in inp for c in row) if inp else set()
            out_colors = set(c for row in out for c in row) if out else set()
            
            color_stats = [
                len(inp_colors), len(out_colors),
                len(inp_colors & out_colors),  # intersection
                len(inp_colors | out_colors),  # union
                max(inp_colors) if inp_colors else 0,
                max(out_colors) if out_colors else 0,
                sum(inp_colors), sum(out_colors)
            ]
            
            # Transformation features (8 features)
            size_preserved = int(ih == oh and iw == ow)
            rotated = int(ih == ow and iw == oh)
            flipped = int(inp == inp[::-1]) if inp else 0
            transposed = int(ih == ow and iw == oh)
            
            # Complexity measures
            inp_complexity = self._measure_complexity(inp)
            out_complexity = self._measure_complexity(out)
            complexity_change = out_complexity - inp_complexity
            
            transform_stats = [
                size_preserved, rotated, flipped, transposed,
                inp_complexity, out_complexity, complexity_change,
                int(self._has_symmetry(inp)) if inp else 0
            ]
            
            features.extend(basic_stats + color_stats + transform_stats)
        
        # Pad or truncate to feature_dim
        features = features[:self.feature_dim] + [0] * max(0, self.feature_dim - len(features))
        return np.array(features, dtype=np.float32)
    
    def _measure_complexity(self, grid: List[List[int]]) -> float:
        """Measure grid complexity using entropy-like metric"""
        if not grid or not grid[0]:
            return 0.0
        
        # Count color frequencies
        color_counts = {}
        total_cells = 0
        
        for row in grid:
            for cell in row:
                color_counts[cell] = color_counts.get(cell, 0) + 1
                total_cells += 1
        
        if total_cells == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in color_counts.values():
            if count > 0:
                prob = count / total_cells
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _has_symmetry(self, grid: List[List[int]]) -> bool:
        """Check if grid has any form of symmetry"""
        if not grid or not grid[0]:
            return False
        
        h, w = len(grid), len(grid[0])
        
        # Horizontal symmetry
        if grid == grid[::-1]:
            return True
        
        # Vertical symmetry
        if all(row == row[::-1] for row in grid):
            return True
        
        # Diagonal symmetry (if square)
        if h == w:
            diagonal_sym = all(
                grid[i][j] == grid[j][i]
                for i in range(h) for j in range(w)
            )
            if diagonal_sym:
                return True
        
        return False


if __name__ == "__main__":
    # Demo usage
    tokenizer = ARCTokenizer(max_tokens=50)
    feature_extractor = FeatureExtractor()
    
    # Example task
    examples = [{
        'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    }]
    
    # Tokenize
    tokens = tokenizer.tokenize_examples(examples)
    decoded = tokenizer.decode_tokens(tokens)
    
    # Extract features
    features = feature_extractor.extract_features(examples)
    
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Features shape: {features.shape}")
    print(f"Feature sample: {features[:10]}")
