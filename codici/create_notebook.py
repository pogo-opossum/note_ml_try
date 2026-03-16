"""
JUPYTER NOTEBOOK GENERATOR

This script converts the regression_improved.py script into a properly formatted
Jupyter notebook (.ipynb) with appropriate cell divisions.

Usage:
    python create_notebook.py input_script.py output_notebook.ipynb

Author: ML Educational Scripts
Date: 2025
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


class NotebookCell:
    """Represents a single Jupyter notebook cell."""
    
    def __init__(self, cell_type: str, content: str):
        """
        Initialize a notebook cell.
        
        Parameters:
        -----------
        cell_type : str
            Type of cell: 'code' or 'markdown'
        content : str
            Cell content
        """
        self.cell_type = cell_type
        self.content = content.strip()
    
    def to_dict(self) -> Dict:
        """Convert cell to dictionary format for JSON serialization."""
        if self.cell_type == 'markdown':
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": self._format_content()
            }
        else:  # code
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": self._format_content()
            }
    
    def _format_content(self) -> List[str]:
        """Format content as list of lines for JSON."""
        lines = self.content.split('\n')
        formatted = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                formatted.append(line + '\n')
            else:
                formatted.append(line)
        return formatted


class PythonToNotebookConverter:
    """Converts Python script with markdown docstrings to Jupyter notebook."""
    
    def __init__(self, script_path: str):
        """
        Initialize converter.
        
        Parameters:
        -----------
        script_path : str
            Path to Python script to convert
        """
        self.script_path = Path(script_path)
        self.content = self.script_path.read_text(encoding='utf-8')
        self.cells: List[NotebookCell] = []
    
    def parse(self) -> None:
        """
        Parse the script and create notebook cells.
        
        Strategy:
        1. Top-level triple-quoted strings (" " ") become markdown cells
        2. Code between strings becomes code cells
        3. Section markers (# ===...) signal new cells
        """
        self._split_by_docstrings()
    
    def _split_by_docstrings(self) -> None:
        """Split script content into docstrings (markdown) and code."""
        # Pattern to match top-level docstrings
        docstring_pattern = r'^\s*"""(.*?)"""'
        
        remaining = self.content
        pos = 0
        
        while pos < len(remaining):
            # Find next docstring
            match = re.search(docstring_pattern, remaining[pos:], re.DOTALL | re.MULTILINE)
            
            if match:
                # Add code before docstring (if any)
                code_before = remaining[pos:pos + match.start()].strip()
                if code_before and not code_before.startswith('"""'):
                    self.cells.append(NotebookCell('code', code_before))
                
                # Add docstring as markdown
                docstring_text = match.group(1).strip()
                self.cells.append(NotebookCell('markdown', docstring_text))
                
                pos += match.end()
            else:
                # No more docstrings, add remaining as code
                remaining_code = remaining[pos:].strip()
                if remaining_code:
                    self.cells.append(NotebookCell('code', remaining_code))
                break
    
    def merge_short_code_cells(self, min_lines: int = 2) -> None:
        """
        Merge consecutive short code cells (useful for better organization).
        
        Parameters:
        -----------
        min_lines : int
            Minimum number of lines to keep cell separate
        """
        merged_cells = []
        i = 0
        
        while i < len(self.cells):
            cell = self.cells[i]
            
            # Check if this is a short code cell
            if cell.cell_type == 'code':
                line_count = len(cell.content.split('\n'))
                
                if line_count < min_lines and i + 1 < len(self.cells):
                    # Try to merge with next code cell
                    if self.cells[i + 1].cell_type == 'code':
                        merged_content = cell.content + '\n\n' + self.cells[i + 1].content
                        merged_cells.append(NotebookCell('code', merged_content))
                        i += 2
                        continue
            
            merged_cells.append(cell)
            i += 1
        
        self.cells = merged_cells
    
    def optimize_cells(self) -> None:
        """
        Optimize cell organization:
        1. Remove empty cells
        2. Split overly long code cells at section markers
        """
        # Remove empty cells
        self.cells = [c for c in self.cells if c.content]
        
        # Split long code cells at section markers
        optimized = []
        for cell in self.cells:
            if cell.cell_type == 'code':
                sub_cells = self._split_code_cell_at_sections(cell.content)
                optimized.extend([NotebookCell('code', sc) for sc in sub_cells])
            else:
                optimized.append(cell)
        
        self.cells = optimized
    
    def _split_code_cell_at_sections(self, code: str) -> List[str]:
        """
        Split code at section markers.
        
        Section markers are lines like:
            # ============================================================================
            # SECTION N: TOPIC
            # ============================================================================
        """
        lines = code.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            # Check if line is a section marker
            if ('# ============================================================================' in line or
                line.strip().startswith('# SECTION') or
                line.strip().startswith('# SUBSECTION')):
                
                if current_section:
                    sections.append('\n'.join(current_section).strip())
                    current_section = []
            
            current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section).strip())
        
        return [s for s in sections if s]  # Remove empty sections
    
    def add_front_matter(self) -> None:
        """Add initial markdown cell with notebook metadata."""
        front_matter = """# Housing Price Regression Analysis

This notebook provides a comprehensive educational exploration of linear regression 
and related techniques for predicting housing prices.

## Notebook Structure

1. **Setup & Data Loading**: Import libraries and load the Boston Housing dataset
2. **Exploratory Data Analysis**: Visualize data distributions and correlations
3. **Single-Feature Regression**: Understand regression on individual features
4. **Multi-Feature Regression**: Extend to multiple input features
5. **Regularization Methods**: Explore L1, L2, and Elastic Net regularization
6. **Polynomial Features**: Non-linear regression with polynomial basis functions
7. **Hyperparameter Tuning**: Optimize models using grid search and cross-validation
8. **Model Comparison**: Compare different approaches and select best model

## Key Concepts Covered

- Linear regression fundamentals and cost functions
- Train/test splitting and cross-validation
- Feature standardization and preprocessing
- Regularization techniques (LASSO, Ridge, Elastic Net)
- Polynomial basis functions
- Grid search for hyperparameter optimization
- Model evaluation metrics (MSE, RMSE, R²)
- Feature selection and interpretability

## Requirements

```python
numpy, pandas, scipy, scikit-learn, matplotlib, seaborn
```

Run the following in the first cell to set up:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```
"""
        self.cells.insert(0, NotebookCell('markdown', front_matter))
    
    def create_notebook(self) -> Dict:
        """Create the notebook dictionary structure."""
        notebook = {
            "cells": [cell.to_dict() for cell in self.cells],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        return notebook
    
    def save(self, output_path: str) -> None:
        """
        Save notebook to file.
        
        Parameters:
        -----------
        output_path : str
            Path where to save the .ipynb file
        """
        output_file = Path(output_path)
        notebook = self.create_notebook()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        print(f"✓ Notebook saved to: {output_file}")
        print(f"  Total cells: {len(self.cells)}")
        
        # Statistics
        code_cells = sum(1 for c in self.cells if c.cell_type == 'code')
        markdown_cells = sum(1 for c in self.cells if c.cell_type == 'markdown')
        print(f"  Code cells: {code_cells}")
        print(f"  Markdown cells: {markdown_cells}")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_notebook.py input_script.py [output_notebook.ipynb]")
        print("\nExample:")
        print("  python create_notebook.py regression_improved.py regression.ipynb")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.ipynb"
    
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Converting {input_path} to notebook format...")
    
    # Create converter and process
    converter = PythonToNotebookConverter(input_path)
    converter.parse()
    converter.optimize_cells()
    converter.merge_short_code_cells(min_lines=3)
    converter.add_front_matter()
    converter.save(output_path)
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
