# CHANGELOG

## 0.6

### Poetry Detection
- Update span-level evalaution
  - Supports unlabeled annotations
  - Supports sytem comparisons
  - Now include page-level F1 scores
  - Updated edge cases for precision and recall, returns 1 when denominator is 0
- Update `get_passim_results` script so span "ref_id" field is renamed to "poem_id"

## [0.5] 2026-05-12

Updates to support publication of PPA found poems v0.5 dataset

### Poetry Detection 
- Logic for managing poetry detection configuration for reference corpora 
  and found poems dataset compilation; a sample config file is provided
  - Compile dataset script uses config file to combine excerpts and 
    metadata (poem and PPA) for publishable dataset
  - `merge_excerpts` method now combines any excerpts with matching
    spans in PPA text; conflicting poem ids are preserved in `alt_poem_ids`
  - Poem metadata now supports optional `cluster_id` for known duplicates
- ReferenceCorpus classes for a consistent way to access reference corpora
  metadata and text
- New utility methods for working with PPA metadata when compiling or loading excerpt data
  in `corppa.poetry_dection.ppa_works`
- Now supports building text corpus from tar.gz file in addition to directory of text files
- Update `run_passim` script to use same corppa-specific defaults for command line and `run_passim` method
- Includes new marimo notebooks for exploring found poems excerpt data

### Misc
- Now supports and tested against both Python 3.12 and 3.13; dropped support for 3.11
- Now using uv for GitHub Actions and local development
- Added pre-commit hook to validate GitHub Actions workflow files
- Sphinx code documentation split out into files matching python modules 
- Experimental scripts to subset excerpt data and upload to Grist 
- Configured intentionally untested code to be exempted from code coverage (CH TML parsing)

## [0.4] 2025-04-30
- Now supports and tested against both Python 3.11 and 3.12
- Now licensed under Apache 2
### Documentation
- Set up Sphinx documentation
- New documentation page specifically for Ends of Prosody participants
- Improved code documentation throughout the code
- Tutorial notebooks by @WHaverals to provide orientation to PPA data and related corppa functionality
- Split out developer documentation from main README
### Annotation
- Prodigy custom command recipe for reporting on annotation progress
- `process_adjudication_data` script to process reviewed annotation data
### Poetry Detection
- Code for parsing and reporting on tags/attributes in Chadwyck-Healey poetry corpus
- New `dataclasses` for `Span`, `Excerpt`, and `LabeledExcerpt`
- Evaluation code and documentation for comparing spans
- `merge_excerpts` script for combining labeled and unlabeled poem excerpts
- Code for working with passim (preparing corpus input files, running passim, working with the results)
- Polars utility methods for working with excerpt data
- `refmatcha` script for identifying excerpts based on matches in local reference corpora (preliminary)
- Preliminary Jupyter notebooks for reviewing found poetry excerpt data
- Preliminary configuration handling for found poetry and reference corpora
### Utilities
- `collate_txt` script to create work-level text corpora files after running OCR
- `build_text_corpus` script to convert a directory of text files into a JSONL corpus
- `get_ppa_source` method now supports all PPA sources (added support for EEBO-TCP)
- New utility function for extracting the page number from the filename of page-level content (e.g., text or image) (currently Gale/ECCO only)
- New utility function that returns a relative path generator of files with one or more extensions under a specified base directory
### Misc
- Added GitHub Actions workflow to check Jupyter notebooks
- Renamed the GitHub repository from `ppa-nlp` to `corppa`; early experimental work not included
  in this package preserved in https://github.com/Princeton-CDH/ppa-nlp-archive
- Increased use of Python type hinting
- Configured codecov with separate reporting for tests and whole project, with different targets for coverage

## [0.3] 2024-11-01
- New dependency: intspan
### Poetry Detection
- New Prodigy recipe for adjudicating text annotations
- Refactored recipes to use Prodigy API
- Extended recipes to optionally fetch media (i.e., images)
- Added unit testing
### Misc
- Fixed Codecov integration

## [0.2] 2024-10-07
- Now requires Python 3.12
### Corppa Utilities
- Basic readme documentation for filter script
- New script for OCR with google vision
- Updated filter script:
  - Uses PPA work ids instead of source ids
  - Additional filtering by volume and page
  - Additional filtering by include or exclude key-pair values
- New utilities function for working with PPA corpus file paths
- New script for generating PPA page subset to be used in conjunction with the filter script
- New script for adding image relative paths to a PPA text corpus
### Poetry Detection
- New Prodigy recipes and custom CSS for image and text annotation
- Script to add PPA work-level metadata for display in Prodigy
### Misc
- Ruff precommit hook now configured to autofix import order


## [0.1] 2024-06-05
- Utility to filter the full text corpus by source ID
- Experimental Scripts
  - OCR evaluation
  - Character-level statistics
  

[0.1]: https://github.com/Princeton-CDH/corppa/releases/tag/0.1
[0.2]: https://github.com/Princeton-CDH/corppa/releases/tag/0.2
[0.3]: https://github.com/Princeton-CDH/corppa/releases/tag/0.3
[0.4]: https://github.com/Princeton-CDH/corppa/releases/tag/0.4
[0.5]: https://github.com/Princeton-CDH/corppa/releases/tag/0.5
