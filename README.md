# corppa

This repository is research software developed as part of the [Ends of Prosody](https://cdh.princeton.edu/projects/the-ends-of-prosody/), which is associated with the [Princeton Prosody Archive](https://prosody.princeton.edu/) (PPA). This software is particularly focused on research and work related to PPA full-text and page image corpora.

> [!WARNING]
> This code is primarily for internal team use. Some portions of it may eventually be useful for participants of the [Ends of Prosody conference](https://cdh.princeton.edu/events/the-ends-of-prosody/) or be adapted or used elsewhere.

## Basic Usage

### Installation

Use pip to install as a python package directly from GitHub.  Use a branch or tag name, e.g. `@develop` or `@0.1` if you need to install a specific version.

```sh
pip install git+https://github.com/Princeton-CDH/ppa-nlp.git#egg=corppa
```
or
```sh
pip install git+https://github.com/Princeton-CDH/ppa-nlp.git@v0.1#egg=corppa
```

### Scripts

Installing `corppa` currently provides access to two command line scripts, for filtering a PPA page-level corpus or for generating OCR text for images using Google Vision API. These can be run as `corppa-filter` and `corppa-ocr` respectively.

#### Filtering PPA page-text corpus

The PPA page-level text corpus is shared as a json lines (`.jsonl`) file, which may or may not be compressed (e.g., `.jsonl.gz`). It's often useful to filter the full corpus to a subset of pages for a specific task, e.g. to analyze content from specific volumes or select particular pages for annotation.

To create a subset corpus with _all pages_ for a set of specific volumes, create a text file with a list of **PPA work identifiers**, one id per line, and then run the filter script with the input file, desired output file, and path to id file.

```sh
corppa-filter ppa_pages.jsonl my_subset.jsonl --idfile my_ids.txt
```

> [!NOTE]
> **PPA work identifiers** are based on source identifiers, i.e., the identifier from the original source (HathiTrust, Gale/ECCO, EEBO-TCP). In most cases the work identifier and the source identifier are the same, but _if you are working with any excerpted content the work id is NOT the same as the source identifier_. Excerpt ids are based on the combination of source identifier and the first original page included in the excerpt. In some cases PPA contains multiple excerpts from the same source, so this provides guaranteed unique work ids.

To create a subset of _specific pages_ from specific volumes, create a CSV file that includes fields `work_id` and `page_num`, and pass that to the filter script with the `--pg-file` option:

```sh
corppa-filter ppa_pages.jsonl my_subset.jsonl --pg_file my_work_pages.csv
```

You can filter a page corpus to exclude or include pages based on exact-matches for attributes included in the jsonl data. For example, to get all pages with the original page number roman numeral 'i':

```sh
corppa-filter ppa_pages.jsonl i_pages.jsonl --include label=i
```

Filters can also be combined; for example, to get the original page 10 for every volume from a list, you could specify a list of ids and the `--include` filter:

```sh
corppa-filter ppa_pages.jsonl my_subset_page10.jsonl --idfile my_ids.txt --include label=10
```


## Development instructions

This repo uses [git-flow](https://github.com/nvie/gitflow) branching conventions; **main** contains the most recent release, and work in progress will be on the **develop** branch. Pull requests for new features should be made against develop.

### Developer setup and installation

- **Recommended:** create a python virtual environment with your tool of choice (virtualenv, conda, etc); use python 3.12 or higher

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
`