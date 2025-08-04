#!/usr/bin/env python3
"""
Add author information to all Python files
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

import os
import re
from pathlib import Path

def add_author_info_to_file(filepath):
    """Add author information header to a Python file."""
    
    # Skip already processed files
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'Author: Muzan Sano' in content:
            return False, "Already has author info"
    
    # Template header
    header = '''"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

'''
    
    lines = content.split('\n')
    
    # Find insertion point (after shebang and initial docstring if present)
    insert_idx = 0
    
    # Skip shebang
    if lines and lines[0].startswith('#!'):
        insert_idx = 1
    
    # Skip existing docstring
    in_docstring = False
    docstring_quotes = None
    
    for i in range(insert_idx, len(lines)):
        line = lines[i].strip()
        
        if not line or line.startswith('#'):
            continue
            
        # Check for docstring start
        if line.startswith('"""') or line.startswith("'''"):
            if not in_docstring:
                docstring_quotes = line[:3]
                in_docstring = True
                if line.count(docstring_quotes) >= 2:  # Single line docstring
                    insert_idx = i + 1
                    break
            elif line.endswith(docstring_quotes):
                insert_idx = i + 1
                break
        elif in_docstring and docstring_quotes and line.endswith(docstring_quotes):
            insert_idx = i + 1
            break
        elif not in_docstring:
            insert_idx = i
            break
    
    # Insert header
    lines.insert(insert_idx, header.rstrip())
    
    # Remove emojis from content (but preserve code examples)
    emoji_pattern = re.compile(r'[^\w\s\.,;:!?\'"()\[\]{}<>+=\-*/\\|&^%$#@~`\n\r\t]', re.UNICODE)
    new_content = '\n'.join(lines)
    
    # Only remove emojis from comments, not from strings that might be code examples
    def remove_emojis_from_comments(match):
        line = match.group(0)
        if line.strip().startswith('#'):
            return emoji_pattern.sub('', line)
        return line
    
    # Apply emoji removal carefully
    new_content = re.sub(r'^.*$', remove_emojis_from_comments, new_content, flags=re.MULTILINE)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True, "Added author info and removed emojis"

def main():
    """Process all Python files in the project."""
    project_root = Path(__file__).parent.parent
    
    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(project_root.glob(pattern))
    
    processed = 0
    skipped = 0
    
    for filepath in python_files:
        # Skip certain directories
        if any(part in str(filepath) for part in ['.git', '__pycache__', 'venv', '.venv']):
            continue
            
        try:
            updated, message = add_author_info_to_file(filepath)
            if updated:
                print(f"✓ {filepath.relative_to(project_root)}: {message}")
                processed += 1
            else:
                print(f"- {filepath.relative_to(project_root)}: {message}")
                skipped += 1
        except Exception as e:
            print(f"✗ {filepath.relative_to(project_root)}: Error - {e}")
    
    print(f"\nSummary: {processed} files updated, {skipped} files skipped")

if __name__ == "__main__":
    main()
