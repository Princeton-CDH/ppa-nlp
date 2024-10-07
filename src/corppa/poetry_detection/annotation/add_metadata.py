"""
This script is used to prep ppa corpus page jsonl data for use in
Prodigy annotation. It adds work metadata (title, author, year) in the
location that Prodigy requires for display, and allows adjusting image
paths for display from the Prodigy interface.  It assumes the input page
corpus has already been annotated with image paths with an `image_path`
attribute, and adds an optional url prefix and converts any .TIF extensions
to .jpg.
"""

# NOTE: this script is provisional and should likely be refactored
# or combined with other page data handling scripts

import argparse
import csv
import pathlib
import sys
from typing import Iterator

import orjsonl
from tqdm import tqdm


def combine_data(
    jsonl_path: pathlib.Path, csv_path: pathlib.Path, disable_progress: bool = False
) -> Iterator[dict]:
    # add work-level metadata to jsonl page data

    # load metadata from csv file
    # - preserve title, author, and publication year
    with csv_path.open() as csvfile:
        csvreader = csv.DictReader(csvfile)
        # create a lookup keyed on work id
        metadata = {
            row["work_id"]: {
                "title": row["title"],
                "author": row["author"],
                "year": row["pub_year"],
            }
            for row in csvreader
        }

    # use orjsonl to open and stream the json lines;
    # wrap in tqdm for optional progress bar
    progress_pages = tqdm(
        orjsonl.stream(jsonl_path),
        disable=disable_progress,
    )

    for page in progress_pages:
        # add metadata dictionary for Prodigy
        page["meta"] = metadata[page["work_id"]]

        yield page


def main():
    """Add work metadata to pages for display in Prodigy"""
    # NOTE: if we decide to keep this script, we should add a named entry
    # for it in the pyproject so it can be run when the package is installed

    parser = argparse.ArgumentParser(
        description="Add PPA work-level metadata to pages for context in Prodigy",
    )
    parser.add_argument(
        "input",
        help="Path to a PPA page-level corpus JSONL file (compressed or not)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "metadata", help="Path to PPA work-level metatada CSV file", type=pathlib.Path
    )
    parser.add_argument(
        "output",
        help="Filename where the updated corpus should be saved",
        type=pathlib.Path,
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

    # input file and metadata files should exist and not be empty
    for input_file in [args.input, args.metadata]:
        if not input_file.exists():
            print(f"Error: {input_file} does not exist")
            sys.exit(-1)
        elif args.input.stat().st_size == 0:
            print(f"Error: {input_file} is zero size")
            sys.exit(-1)

    # output file should not exist
    if args.output.exists():
        print(f"Error: output file {args.output} already exists, not overwriting")
        sys.exit(-1)

    # use orjsonl to stream updated pages to specified output file
    orjsonl.save(
        args.output,
        combine_data(args.input, args.metadata, disable_progress=disable_progress),
    )


if __name__ == "__main__":
    main()
