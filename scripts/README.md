# Experimental Scripts
This directory contains stand-alone, experimental scripts associated with the `corppa` package.

## Setup and Installation
Scripts may require their own specific environment.
These are specified as conda `.yml` files and are located within `scripts/envs`.

For example, to create and activate the `ppa-ocr` environment, run the following commands:
```
conda env create -f scripts/envs/ppa-ocr.yml
conda activate ppa-ocr
```

## Script Descriptions

### get_character_stats.py
This script compiles character-level statistics for a PPA text (sub)corpus.
It requires the `ppa-ocr` environment.
This script has an additional dependency, it requires the [en_core_web_lg](https://spacy.io/models/en#en_core_web_lg) spacy language model, which can be downloaded by running the following command:
```
python -m spacy download en_core_web_lg
```

#### evaluate_ocr.py
This script compiles OCR quality statistics for a PPA text (sub)corpus.
It requires the `ppa-ocr` environment.

## Helper Modules
These modules contain auxiliary methods that stand-alone scripts may use.
Check dependencies before using modules elsewhere.

#### `helper.py`
This module contains general-purpose auxiliary methods.

#### `ocr_helper.py`
This module contains OCR-related auxiliary methods.

### `transform-images.sh`
This bash script will copy and transform images from a PPA (sub)corpus (jsonl).
