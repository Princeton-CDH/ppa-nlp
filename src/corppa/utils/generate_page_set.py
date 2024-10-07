"""
Utility for generating a PPA page set.

This method takes three inputs: (1) an input csv, (2) an output csv, and
(3) the size of the page set.

The input CSV file must have the following fields:
    * work_id: PPA work id
    * page_start: Starting index for page range being considered for this work
    * page_end: Ending index for page range being considered for this work
    * poery_pages: Comma separated list of page numbers containing poetry

The pages are selected as follows:
    * First, all pages with poetry are selected
    * Then, all remaining pages are chosen randomly (proportionately by work)

The resulting output CSV file has the following fields:
    * work_id: PPA work id
    * page_num: Digital page number
"""

import argparse
import csv
import random
import sys
from pathlib import Path


def get_pages(in_csv, k):
    """
    Using the input CSV, generate a page set with k pages such that all pages
    with poetry are included and the remaining pages are selected randomly from
    the remaining pages under consideration.

    Notes:
        * In some cases the generated page set may not match k.
        * This is not compatible with PPA works with non-sequential page ranges.
    """
    # Load page data
    page_pool = {}
    poetry_pages = {}
    page_counter = 0
    with open(in_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            work_id = row["work_id"]
            if work_id not in page_pool:
                page_pool[work_id] = {}
            start_idx = int(row["page_start"])
            end_idx = int(row["page_end"]) + 1
            # Gather full page range
            for i in range(start_idx, end_idx):
                page_pool[work_id][i] = {"work_id": work_id, "page_num": i}
            # Yield pages with poetry
            for pg_id in row["poetry_pages"].split(","):
                yield page_pool[work_id].pop(int(pg_id))
                page_counter += 1

    # Print warning if more poetry pages than k
    if page_counter >= k:
        print(
            f"Warning: Too many pages with poetry (k = {k} <= {page_counter})",
            file=sys.stderr,
        )

    # Select remaining pages randomly
    # TODO: Revisit to simply page selcection logic
    while page_counter < k:
        # Select work
        work_id = random.choice(list(page_pool.keys()))
        # Select page
        try:
            pg_id = random.choice(list(page_pool[work_id].keys()))
        except IndexError:
            # Encountered empty list, remove work entry and continue
            del page_pool[work_id]
            continue
        yield page_pool[work_id].pop(pg_id)
        page_counter += 1

    # Print warning if less than k pages found
    if page_counter < k:
        print(f"Warning: Less than k pages found", file=sys.stderr)


def save_page_set(in_csv, out_csv, k):
    """
    Save a page set of size k constructed based on the input csv
    """
    with open(out_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["work_id", "page_num"])
        writer.writeheader()
        for page in get_pages(in_csv, k):
            writer.writerow(page)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a page set suitable for filtering PPA"
    )
    # Required Arguments
    parser.add_argument(
        "input",
        help="Input CSV",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Output CSV",
        type=Path,
    )
    parser.add_argument("k", help="Number of pages in set", type=int)

    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: input {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    if args.output.is_file():
        print(f"Error: output {args.output} already exists", file=sys.stderr)
        sys.exit(1)
    if args.k <= 0:
        print(f"Error: k must be positive", file=sys.stderr)
        sys.exit(1)

    save_page_set(args.input, args.output, args.k)


if __name__ == "__main__":
    main()
