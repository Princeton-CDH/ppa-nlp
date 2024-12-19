"""
Gather excerpts from passim page-level results.
"""

import argparse
import csv
import pathlib
import sys

import orjsonl


def get_passage_matches(input_file):
    """
    Extracts and yields the passim-identified passage-level matches
    """
    for page in orjsonl.stream(input_file):
        if not page["n_spans"]:
            # Skipe pages without matches
            continue
        for poem_span in page["poem_spans"]:
            match = poem_span.copy()
            match["page_id"] = page["page_id"]
            yield match


def save_passim_passage_matches(input_file, output_file):
    with open(output_file, mode="w", newline="") as tsvfile:
        fieldnames = [
            "page_id",
            "ref_id",
            "ref_corpus",
            "page_start",
            "page_end",
            "ref_start",
            "ref_end",
            "page_excerpt",
            "ref_excerpt",
            "aligned_page_excerpt",
            "aligned_ref_excerpt",
        ]
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, dialect="excel-tab")
        writer.writeheader()
        for match in get_passage_matches(input_file):
            writer.writerow(match)


def main():
    """
    Command-line access to build TSV file of passage-level passim matches.
    """
    parser = argparse.ArgumentParser(
        description="Extract passage-level passim results (TSV)"
    )

    # Required arguments
    parser.add_argument(
        "input",
        help="Page-level passim results file (JSONL)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Filename for passage-level passim output (TSV)",
        type=pathlib.Path,
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input.is_file():
        print(f"Error: input {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    if args.output.is_file():
        print(f"Error: output file {args.output} exist", file=sys.stderr)
        sys.exit(1)

    save_passim_passage_matches(args.input, args.output)


if __name__ == "__main__":
    main()
