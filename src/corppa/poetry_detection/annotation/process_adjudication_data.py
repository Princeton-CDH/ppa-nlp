"""
This script processes the adjudication data produced by Prodigy for our
poetry detection task into two outputs:

    1. A JSONL file that compiles the annotation data into page-level records.
       So, each record contains some page-level metdata and the compiled list
       of poetry excerpts (if any) determined in the adjudication process.

    2. A CSV file containing excerpt-level data per line.

Note that the first file explicitly include information on the pages where
no poetry was identified, while the second will only implicitly through
absence and requires external knowledge of what pages were covered in
the annotation rounds. So, the former is particularly useful for the evaluation
process while the latter is better suited for building a final excerpt dataset.

Example command line usage:
```
python process_adjudication_data.py prodigy_data.jsonl adj_pages.jsonl adj_excerpts.csv
```
"""

import argparse
import csv
import pathlib
import sys
from collections.abc import Generator
from typing import Any

import orjsonl
from tqdm import tqdm
from xopen import xopen


def get_excerpts(page_annotation: dict[str, Any]) -> list[dict[str, int | str]]:
    """
    Extract excerpts from page-level annotation. Excerpts have the following
    fields:
        * start: character-level starting index
        * end: character-level end index (Pythonic, exclusive)
        * text: text of page excerpt

    Note: Currently ignoring span labels, since there's only one for the
          poetry detection task.
    """
    excerpts = []
    # Blank pages may not have a text field, so in these cases set to empty string
    page_text = page_annotation.get("text", "")
    if "spans" not in page_annotation:
        raise ValueError("Page annotation missing 'spans' field")
    for span in page_annotation["spans"]:
        excerpt = {
            "start": span["start"],
            "end": span["end"],
            "text": page_text[span["start"] : span["end"]],
        }
        excerpts.append(excerpt)
    return excerpts


def process_page_annotation(page_annotation) -> dict[str, Any]:
    """
    Extracts desired content from page-level annotation. The returned data has
    the following fields"
        * page_id: Page's PPA page identifier
        * work_id: PPA work identifier
        * work_title: Title of PPA work
        * work_author: Author of PPA work
        * work_year: Publication of PPA work
        * n_excerpts: Number of poetry excerpts contained in page
        * excerpts: List of poetry excerpts identified within page
    """
    page_data = {}
    page_data["page_id"] = page_annotation["id"]
    page_data["work_id"] = page_annotation["work_id"]
    page_data["work_title"] = page_annotation["meta"]["title"]
    page_data["work_author"] = page_annotation["meta"]["author"]
    page_data["work_year"] = page_annotation["meta"]["year"]
    page_data["excerpts"] = get_excerpts(page_annotation)
    page_data["n_excerpts"] = len(page_data["excerpts"])
    return page_data


def get_excerpt_entries(page_data: dict[str, Any]) -> Generator[dict[str, Any]]:
    """
    Generate excerpt entries data from the processed page produced by
    `process_page_annotation`.
    """
    for excerpt in page_data["excerpts"]:
        entry = {
            "page_id": page_data["page_id"],
            "work_id": page_data["work_id"],
            "work_title": page_data["work_title"],
            "work_author": page_data["work_author"],
            "work_year": page_data["work_year"],
            "start": excerpt["start"],
            "end": excerpt["end"],
            "text": excerpt["text"],
        }
        yield entry


def process_adjudication_data(
    input_jsonl: pathlib.Path,
    output_pages: pathlib.Path,
    output_excerpts: pathlib.Path,
    disable_progress: bool = False,
) -> None:
    """
    Process adjudication annotation data and write output files containing page-level
    and excerpt-level information that are JSONL and CSV files respectively.
    """
    n_lines = sum(1 for line in xopen(input_jsonl, mode="rb"))
    progress_annos = tqdm(
        orjsonl.stream(input_jsonl),
        total=n_lines,
        disable=disable_progress,
    )
    csv_fieldnames = [
        "page_id",
        "work_id",
        "work_title",
        "work_author",
        "work_year",
        "start",
        "end",
        "text",
    ]
    with open(output_excerpts, mode="w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        csv_writer.writeheader()
        for page_anno in progress_annos:
            # Get & save page data
            page_data = process_page_annotation(page_anno)
            orjsonl.append(output_pages, page_data)

            for row in get_excerpt_entries(page_data):
                csv_writer.writerow(row)


def main():
    """
    Extracts page- and excerpt-level data from a Prodigy data file (JSONL)
    and writes the page-level excerpt data to a JSONL (`output_pages`) and the
    excerpt-level data to a CSV (`output_excerpts`).
    """
    parser = argparse.ArgumentParser(
        description="Extracts & saves page- and excerpt-level data from Prodigy data file",
    )
    parser.add_argument(
        "input",
        help="Path to Prodigy annotation data export (JSONL file)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output_pages",
        help="Filename where extracted page-level data (JSONL file) should be written",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output_excerpts",
        help="Filename where extracted excerpt-level data (CSV file) should be written",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    disable_progress = not args.progress

    # Check that input file exists
    if not args.input.is_file():
        print(
            f"Error: input file {args.input.is_file()} does not exist", file=sys.stderr
        )
        sys.exit(1)

    # Check that output files does not exist
    for output_file in [args.output_pages, args.output_excerpts]:
        if output_file.exists():
            print(
                f"Error: output file {output_file} already exists, not overwriting",
                file=sys.stderr,
            )
            sys.exit(1)

    process_adjudication_data(
        args.input,
        args.output_pages,
        args.output_excerpts,
        disable_progress=disable_progress,
    )


if __name__ == "__main__":
    main()
