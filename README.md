# `corppa`  PPA full-text corpus utilities

This repository provides code and other resources associated with the [Princeton Prosody Archive](https://prosody.princeton.edu/) (PPA), with a  particular focus on working with the PPA full-text corpus.

## Development instructions

This repo uses [git-flow](https://github.com/nvie/gitflow) branching conventions; **main** contains the most recent release, and work in progress will be on the **develop** branch. Pull requests for new features should be made against develop.

### Developer setup and installation

- **Recommended:** create a python virtual environment with your tool of choice (virtualenv, conda, etc); use python 3.10 or higher

- Install the local checked out version of this package in editable mode (`-e`), including all python dependencies  and optional dependencies for development and testing:
```sh
pip install -e ".[dev]"
```

- This repository uses [pre-commit](https://pre-commit.com/) for python code linting and consistent formatting. Run this command to initialize and install pre-commit hooks:
```sh
pre-commit install
```

## Experimental Scripts
Experimenatal scripts are located within the `scripts` directory.

### Setup and installation
Scripts may require their own specific environment.
These are specified as conda `.yml` files and are located within `scripts/envs`.

For example, to create and activate the `ppa-ocr` environment, run the following commands:
```
conda env create -f scripts/envs/ppa-ocr.yml
conda activate ppa-ocr
```

### Script descriptions
- `get_character_stats.py`: Compiles character-level statistics for a PPA text (sub)corpus
  - Environment: `ppa-ocr`
  - Download Spacy English language model: `python -m spacy download en_core_web_lg`
- `evaluate_ocr.py`: Compiles OCR quality statistics for a PPA text (sub)corpus
  - Environment: `ppa-ocr`
