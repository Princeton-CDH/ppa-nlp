# Passim Pipeline
This directory contains the various scripts needed for running Passim
to detect the reuse of poetry on pages from works in PPA.

> [!WARNING]
> This workflow is quite preliminary and may undergo extensive change.

## Preliminary Passim Workflow

### 1. Create input corpora for `passim`
The pipeline requires one or more corpus JSONL files formatted for Passim.
The script `create_passim_corpus.py` will transform a given text corpus
JSONL file into a new format suitable for passing to Passim.

Minimally, for each record (i.e. line) the follow three fields are created:
1. `id`: The working id for the text record (e.g. ppa page or reference poem id).
         This is derived from the input record's `id` field. Optionally, this can
         be set to an alternative record field.
2. `corpus`: The corpus name the text record belongs to. This is provided as
             input to `create_passim_corpus.py`.
3. `text`: The record's text derived from the input record's `text` field.

Optionally, all other fields of the input records may be preserved by including the
`--preserve-fields` flag.

Example usage:
```
python create_passim_corpus.py ppa_pages.jsonl ppa-pages-passim.jsonl ppa
python create_passim_corpus.py poems.jsonl ref-poems.jsonl ref-poems
```

### 2. Run `passim` using its default settings
Then, we can run `passim` using the `run_passim.sh` script. Note that passim requires
Java 8\*/11/17. This script has two input arguments:
1. Input paths that, following Spark conventions, can be a single file, directory,
or `*` wildcard expression. Multiple paths can be provided by enclosing the
comma-separated list of paths in curly braces (e.g. `"{path1,path2}"`)
2. A directory path in which `passim`'s output will be written. If the directory does
not exist, it will be created.

Note that this will compare PPA texts (`corpus = "ppa"`) with texts from other sources
(`corpus != "ppa"`)

Example usage:
```
./run_passim.sh passim-input.jsonl passim-output
env PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH" ./run_passim.sh "input/*.jsonl" passim-output
./run_passim.sh "{input/ppa_text.jsonl,input/ref_text.jsonl}" passim-output
```

### 3. Build page-level results from `passim` output
After running `passim`, we can use the `get_passim_page_results.py` script to build
the page-level passages identified by `passim` in a JSONL file. This will include records
for pages where `passim` identifies *no* reuse.

The intended use cases of this output are two-fold:
1. Analysing and evaluting the performance of passim
2. Extracting the excerpts identified by passim

Currently, this script has two effective modes. One which includes the underlying text
excerpts and one that does not. This is due to the fact that `passim` output files do not
contain the full excerpts themselves, only an aligned version (one that includes "-" symbols
to indicate places where a character was "inserted" for alignment).

#### Get page-level `passim` results without excerpts
Currently, the default case does not include excerpts in the output JSONL file. It requires
the following arguments:
- `input_corpus`: The input corpus containing PPA texts used in in the passim run
- `passim_output_dir`: The passim output directory for the passim run
- `output`: Filename for the output page-level results (JSONL)

Example usage:
```
python get_passim_page_results.py ppa_passim.jsonl passim-output passim_page_results.jsonl
```

#### Get page-level `passim` results with excerpts
To include the full excerpt texts, the `--include-excerpts` flag must be set. Additionally,
any reference corpus files (JSONL) used in the passim run must be provided using the
`--ref-corpus` optional argument.

Example usage:
```
python get_passim_page_results.py ppa_text.jsonl passim-output passim_page_results.jsonl \
   --include-excerpts --ref-corpus ref_poems.jsonl
```

### 4. Build excerpt-level results
Finally, an excerpt-level results file (TSV) can be built from the page-level results
*with excerpts* built in the previous step using `get_passim_excerpts.py`.

```
python get_passim_excerpts.py passim_page_results.jsonl passim_excerpts.tsv
```
