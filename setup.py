from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='minimol',
    version='0.2.2',
    packages=find_packages(),
    install_requires=[
        'typer',
        'loguru',
        'omegaconf>=2.0.0',
        'tqdm',
        'platformdirs',
        'numpy',
        'scipy>=1.4',
        'pandas>=1.0',
        'scikit-learn',
        'fastparquet',
        'matplotlib>=3.0.1',
        'seaborn',
        'fsspec>=2021.6',
        's3fs>=2021.6',
        'gcsfs>=2021.6',
        'torch>=2.0',
        'lightning>=2.0',
        'torchmetrics>=0.7.0,<0.11',
        'ogb',
        'pytorch_geometric>=2.0',
        'wandb',
        'mup',
        'pytorch_sparse>=0.6',
        'pytorch_cluster>=1.5',
        'pytorch_scatter>=2.0',
        'rdkit',
        'datamol>=0.10',
        'sympy',
        'tensorboard',
        'pydantic<2',
        'pytest>=6.0',
        'pytest-xdist',
        'pytest-cov',
        'pytest-forked',
        'nbconvert',
        'black>=23',
        'jupyterlab',
        'ipywidgets',
        'mkdocs',
        'mkdocs-material',
        'mkdocs-material-extensions',
        'mkdocstrings',
        'mkdocstrings-python',
        'mkdocs-jupyter',
        'markdown-include',
        'mike>=1.0.0',
        'hydra-core>=1.3.2',
    ],
    url='https://github.com/graphcore-research/minimol',
    author='Blazej Banaszewski, Kerstin Klaser',
    author_email='blazej@banaszewski.pl, kerstink@graphcore.ai',
    description='Molecular fingerprinting using pre-trained deep nets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.9',
)
