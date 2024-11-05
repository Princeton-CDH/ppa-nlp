#!/usr/bin/env python
"""
Script to turn directories with multiple text files into a single JSON
file containing text contents of all files with page numbers based
on text filenames. (Page number logic is currently Gale-specific).
"""

import argparse
import json
import pathlib
import sys

from tqdm import tqdm

from corppa.utils.path_utils import find_relative_paths, get_page_number


def collate_txt(
    input_dir: pathlib.Path, output_dir: pathlib.Path, show_progress: bool = True
):
    """Takes a directory that contains text files grouped by directory at any
    level of nesting under the specified `input_dir` and combines them into
    one JSON file per directory. JSON files are created in the specified
    `output_dir` using the same hierarchy found in the `input_dir`.
    """
    directories = 0
    txt_files = 0
    skipped = 0

    # stack tqdm bars so we can briefly show status
    status = tqdm(
        desc="Collating",
        bar_format="{desc}{postfix}",
        disable=not show_progress,
    )

    for ocr_dir, files in tqdm(
        find_relative_paths(input_dir, [".txt"], group_by_dir=True),
        desc="Directories with text files",
        disable=not show_progress,
    ):
        # output will be a json file based on name of the directory containing text files,
        # with parallel directory structure to the source
        output_file = output_dir / ocr_dir.parent / f"{ocr_dir.name}.json"
        # if output exists from a previous run, skip
        if output_file.exists():
            skipped += 1
            continue

        directories += 1
        txt_files += len(files)
        status.set_postfix_str(f" {ocr_dir.stem}: {len(files)} txt files")

        # combine text contents into a dictionary keyed on page number
        txt_data = {}
        for filename in files:
            with (input_dir / filename).open(encoding="utf-8") as txtfile:
                txt_data[get_page_number(filename)] = txtfile.read()

        # ensure the parent directory exists
        output_file.parent.mkdir(exist_ok=True)
        # save out text content as json
        with output_file.open("w", encoding="utf-8") as outfile:
            json.dump(txt_data, outfile)

    status.set_postfix_str("")
    status.close()

    # report a summary of what was done
    print(
        f"\nCreated JSON file{'' if directories == 1 else 's'} for "
        + f"{directories:,} director{'y' if directories == 1 else 'ies'} "
        + f"with {txt_files:,} total text files; skipped {skipped:,}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create JSON files to group OCR text files by directory."
    )
    # Required arguments
    parser.add_argument(
        "input_dir",
        help="Top-level input directory with directories of OCR text files.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output_dir",
        help="Top-level output directory for OCR consolidated into JSON files.",
        type=pathlib.Path,
    )
    # Optional arguments
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    # Validate arguments
    if not args.input_dir.is_dir():
        print(
            f"Error: input directory {args.input_dir} does not exist", file=sys.stderr
        )
        sys.exit(1)
    # create output dir if it doesn't exist
    if not args.output_dir.is_dir():
        try:
            args.output_dir.mkdir()
            print(f"Creating output directory {args.output_dir}")
        except (FileExistsError, FileNotFoundError) as err:
            print(
                f"Error creating output directory {args.output_dir}: {err}",
                file=sys.stderr,
            )
            sys.exit(1)

    collate_txt(args.input_dir, args.output_dir, show_progress=args.progress)


if __name__ == "__main__":
    main()
