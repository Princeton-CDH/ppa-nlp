# CHANGELOG

## 0.3.0
- New dependency: intspan
### Poetry Detection
- New Prodigy recipe for adjudicating text annotations
- Refactored recipes to use Prodigy API
- Extended recipes to optionally fetch media (i.e., images)
- Added unit testing
### Misc
- Fixed Codecov integration

## 0.2.0
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


## 0.1.0
- Utility to filter the full text corpus by source ID
- Experimental Scripts
  - OCR evaluation
  - Character-level statistics
