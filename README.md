# `corppa` PPA full-text corpus utilities

This repository provides code and other resources associated with the [Princeton Prosody Archive](https://prosody.princeton.edu/) (PPA), with a particular focus on working with the PPA full-text corpus.

This code is primarily for internal team use. Some portions of it may eventually be useful for participants at the Ends of Prosody conference or elsewhere.

## Basic Usage

- installation

- scripts, basic usage

--- filter script

-- ocr script

-- others?

## Development instructions

This repo uses [git-flow](https://github.com/nvie/gitflow) branching conventions; **main** contains the most recent release, and work in progress will be on the **develop** branch. Pull requests for new features should be made against develop.

### Developer setup and installation

- **Recommended:** create a python virtual environment with your tool of choice (virtualenv, conda, etc); use python 3.10 or higher

- Install the local checked out version of this package in editable mode (`-e`), including all python dependencies and optional dependencies for development and testing:

```sh
pip install -e ".[dev]"
```

- This repository uses [pre-commit](https://pre-commit.com/) for python code linting and consistent formatting. Run this command to initialize and install pre-commit hooks:

```sh
pre-commit install
```

## Experimental Scripts

Experimental scripts associated with `corppa` are located within the `scripts` directory.
See this directory's README for more detail.
