"""
Utility for filtering PPA full-text corpus to work with a subset of
pages. Currently supports filtering by a list of PPA source ids.

.. Note::
    Currently, there is no way to filter to a specific excerpt when
    there are multiple excerpts from a single source.

Filter methods can be run via command-line or python code. Takes jsonl file
(compressed or not) as input, a filename for output, and a file with a list of
selected source ids.

To use as a command-line script, pass corpus as input, desired output filename,
and filename with the list of source ids:

```
corppa-filter-corpus path/to/ppa_pages.jsonl my_ids.txt output/ppa_subset_pages.jsonl
```

Input format and output filename can use any extension supported by :mod:`orjsonl`,
with or without compression; e.g. `.jsonl`, `.jsonl.gz`, `.jsonl.bz2`, etc.

"""

import argparse
import os.path
from typing import Iterator
import sys

import orjsonl
from orjson import JSONDecodeError
from tqdm import tqdm


def filter_pages(
    input_filename: str, source_ids: list[str], disable_progress: bool = False
) -> Iterator[dict]:
    """Takes a filename for a PPA full-text corpus in a format orjsonl supports
    and a list of source ids. Returns a generator of filtered pages from the
    full corpus corresponding to the list of ids.  Displays progress
    with :mod:`tqdm` progress bar unless disabled.

    :param input_filename: str, filename for corpus input
    :param source_ids: list of str, source ids to include in filtered pages
    :param disable_progress: boolean, disable progress bar (optional, default: False)
    :returns: generator of dict with page data
    :raises: FileNotFoundError, orjson.JSONDecodeError
    """
    # convert list of source ids to set for fast hashmap lookup
    source_ids = set(source_ids)
    selected_pages = 0
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


def save_filtered_corpus(
    input_filename: str,
    idfile: str,
    output_filename: str,
    disable_progress: bool = False,
) -> None:
    """Takes a filename for input PPA full-text corpus in a format
    orjsonl supports, filename where filtered corpus should be saved,
    and a filename with a list of source ids, one id per line.
    Calls :meth:`filter_pages`.

    :param input_filename: str, filename for corpus input
    :param idfile: str, filename for list of source ids
    :param output_filename: str, filename for filtered corpus output
    :param disable_progress: boolean, disable progress bar (optional, default: False)
    """
    # read the id file and generate a list of ids
    with open(idfile) as idfile_content:
        source_ids = [line.strip() for line in idfile_content]

    # use orjsonl to stream filtered pages to specified output file
    orjsonl.save(
        output_filename,
        filter_pages(input_filename, source_ids, disable_progress=disable_progress),
    )


def main():
    """Command-line access to filtering the corpus. Available as
    `corppa-filter-corpus` when this package is installed with pip."""

    parser = argparse.ArgumentParser(
        description="Filters PPA full-text corpus by list of source ids",
    )
    parser.add_argument(
        "input",
        help="PPA full-text corpus to be "
        + "filtered; must be a JSONL file (compressed or not)",
    )
    parser.add_argument("idfile", help="filename with list of source ids, one per line")
    parser.add_argument(
        "output", help="filename where the filtered corpus should be saved"
    )
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    # progress bar is enabled by default; disable if requested
    disable_progress = not args.progress

    if not os.path.exists(args.idfile):
        print(f"Error: idfile {args.idfile} does not exist")
        sys.exit(-1)
    elif os.path.getsize(args.idfile) == 0:
        print(f"Error: idfile {args.idfile} is zero size")
        sys.exit(-1)

    # if requested output filename has no extension, add jsonl
    output_filename = args.output
    if os.path.splitext(output_filename)[1] == "":
        output_filename = f"{output_filename}.jsonl"

    if os.path.exists(output_filename):
        print(
            f"Error: requested output file {args.output} already exists; not overwriting"
        )
        sys.exit(-1)

    try:
        save_filtered_corpus(
            args.input, args.idfile, output_filename, disable_progress=disable_progress
        )
    except (FileNotFoundError, JSONDecodeError) as err:
        # catch known possible errors and display briefly
        # with the type of error and the brief message
        print(f"{err.__class__.__name__}: {err}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
