"""
MicroGolf Package Setup
Ultra-compact ARC-AGI solution framework for NeurIPS 2025 Google Code Golf Championship
"""
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = (Path(__file__).parent / "README.md").read_text()

# Read requirements
requirements = []
req_file = Path(__file__).parent / "requirements.txt"
if req_file.exists():
    requirements = req_file.read_text().strip().split('\n')

setup(
    name="microgolf",
    version="1.0.0",
    author="MicroGolf Team",
    author_email="team@microgolf.ai",
    description="Ultra-compact ARC-AGI solution framework for code golf competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microgolf/neurips2025",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1",
            "pytest-cov>=4.0",
            "black>=23.1",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=2.20",
            "astor>=0.8",
        ],
        "ml": [
            "wandb>=0.15",
            "tensorboard>=2.10",
            "jupyter>=1.0",
            "matplotlib>=3.7",
            "seaborn>=0.12",
        ],
        "full": [
            "pytest>=7.1",
            "pytest-cov>=4.0", 
            "black>=23.1",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=2.20",
            "astor>=0.8",
            "wandb>=0.15",
            "tensorboard>=2.10",
            "jupyter>=1.0",
            "matplotlib>=3.7",
            "seaborn>=0.12",
        ]
    },
    entry_points={
        "console_scripts": [
            "microgolf-train=scripts.train_meta_controller:main",
            "microgolf-eval=scripts.eval_submission:main",
            "microgolf-optimize=scripts.prune_characters:main",
        ],
    },
    include_package_data=True,
    package_data={
        "microgolf": [
            "primitives/*.py",
            "model/checkpoints/*.pt",
        ],
    },
    zip_safe=False,
    keywords=[
        "artificial-intelligence",
        "code-golf", 
        "arc-agi",
        "meta-learning",
        "neurips",
        "competition",
        "code-generation",
        "optimization"
    ],
    project_urls={
        "Bug Reports": "https://github.com/microgolf/neurips2025/issues",
        "Source": "https://github.com/microgolf/neurips2025",
        "Documentation": "https://microgolf.readthedocs.io/",
        "Competition": "https://www.kaggle.com/competitions/google-code-golf-2025",
    },
)
