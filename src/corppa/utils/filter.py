"""
Utility for filtering PPA full-text corpus to work with a subset of
pages. Currently supports filtering by a list of PPA source ids.
Currently, there is no way to filter to a specific excerpt when
there are multiple.

Can be run via command-line or python code. Takes jsonl file (compressed or
not) as input, a filename for output, and a file with a list of
selected source ids.

To use as a command-line script:
```
corppa-filter-corpus path/to/ppa_pages.jsonl output/ppa_subset_pages.jsonl my_ids.txt
```
Input format and output filename can use any extension supported by mod:`orjsonl`,
with or without compression; e.g. `.jsonl`, `.jsonl.gz`, `.jsonl.bz2`, etc.

"""

import argparse

import orjsonl
from tqdm import tqdm


def filter_pages(input_filename, source_ids, disable_progress=False):
    """Takes a filename for a PPA full-text corpus in a format orjsonl supports
    and a list of source ids. Returns a generator of filtered pages from the
    full corpus corresponding to the list of ids.
    """
    # convert list of source ids to set for fast hashmap lookup
    source_ids = set(source_ids)
    selected_pages = 0
    # TODO: handle errors orjson.JSONDecodeError; FileNotFoundError
    progress_pages = tqdm(
        orjsonl.stream(input_filename),
        desc="Filtering",
        bar_format="{desc}: checked {n:,} pages{postfix} | elapsed: {elapsed}",
        disable=disable_progress,
    )
    for page in progress_pages:
        # page data does not include source id, but does include work id
        # which is either source id (for full works) or
        # source id plus first page number (for articles/excerpts)
        if page["work_id"].split("-p")[0] in source_ids:
            # keep track of how many have been selected for reporting in
            # progress bar
            selected_pages += 1
            progress_pages.set_postfix_str(f"selected {selected_pages:,}")
            yield page

    # NOTE: other filters could be implemented here later, e.g.
    # based on HathiTrust page tags like UNTYPICAL_PAGE or text content


def save_filtered_corpus(input_filename, output_filename, idfile):
    """Takes a filename for input PPA full-text corpus in a format
    orjsonl supports, filename where filtered corpus should be saved,
    and a filename with a list of source ids, one id per line.
    """
    # read the id file and generate a list of ids
    with open(idfile) as idfile_content:
        source_ids = [line.strip() for line in idfile_content]

    # use orjsonl to stream filtered pages to specified output file
    orjsonl.save(output_filename, filter_pages(input_filename, source_ids))


def main():
    # command-line access to filtering the corpus
    parser = argparse.ArgumentParser(
        description="Filters PPA full-text corpus by list of source ids",
    )
    parser.add_argument(
        "input",
        help="PPA full-text corpus to be "
        + "filtered; must be a JSONL file (compressed or not)",
    )
    parser.add_argument(
        "output", help="filename where the filtered corpus should be saved"
    )
    parser.add_argument("idfile", help="filename with list of source ids, one per line")

    args = parser.parse_args()
    save_filtered_corpus(args.input, args.output, args.idfile)
