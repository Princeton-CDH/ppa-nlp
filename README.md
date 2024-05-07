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

#### get_character_stats.py
This script compiles character-level statistics for a PPA text (sub)corpus.
It requires the `ppa-ocr` environment.
This script has an additional dependency, it requires the [en_core_web_lg](https://spacy.io/models/en#en_core_web_lg) spacy language model, which can be downloaded by running the following command:
```
python -m spacy download en_core_web_lg
````
#### evaluate_ocr.py
This script compiles OCR quality statistics for a PPA text (sub)corpus
It requiers the `ppa-ocr` environment.
