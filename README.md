# Code base for analyzing data recorded from mice.

## Installation

### Prerequisites

[Conda](https://docs.conda.io/en/latest/) (Anaconda or Miniconda) must be installed.

### 1. Create the conda environment

From the root of the repository:

```bash
conda env create -f environment.yml
conda activate els-project
```

### 2. Install the local package

With the environment active, install the `src` package in editable mode:

```bash
pip install -e .
```

The `-e` flag installs in editable (development) mode, so any changes to the source files in `src/` are reflected immediately without reinstalling.

### 3. Verify

```python
from src import helper  # or any other module in src/
```

### Updating the environment

If `environment.yml` changes (e.g., after pulling new commits):

```bash
conda env update -f environment.yml --prune
```
