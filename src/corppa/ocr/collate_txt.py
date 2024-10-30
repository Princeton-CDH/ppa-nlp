#!/usr/bin/env python
"""
Script to turn directories with multiple text files into a single JSON
file containing text contents of all files with page numbers based
on text filenames. (Page number logic is currently Gale-specific).
"""

import argparse
import csv
import json
import pathlib

from tqdm import tqdm

from corppa.utils.path_utils import find_relative_paths


def page_number(filename: pathlib.Path) -> str:
    # NOTE: this logic is currently specific to Gale/ECCO files,
    # which look like CW0112029406_00180.txt

    # split the file base/stem name by _ and take the last part
    pagenum = filename.stem.split("_")[-1]
    # return the number as a string; strip extra trailing zero
    return pagenum[:-1]  # strip trailing zero


def collate_txt(
    input_dir: pathlib.Path, output_dir: pathlib.Path, show_progress: bool = True
):
    directories = 0
    txt_files = 0
    skipped = 0

    # stack tqdm bars so we can briefly show status
    status = tqdm(
        desc="Collating",
        bar_format="{desc}{postfix}",
        disable=not show_progress,
    )
    csv_fieldnames = ["page_number", "text"]

    for ocr_dir, files in tqdm(
        find_relative_paths(input_dir, [".txt"], group_by_dir=True),
        desc="Directories with text files",
        disable=not show_progress,
    ):
        # output will be a json file based on name of the directory containing text files,
        # with parallel directory structure to the source
        output_file = (output_dir / ocr_dir.parent / ocr_dir.name).with_suffix(".csv")
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
            with (input_dir / filename).open() as txtfile:
                txt_data[page_number(filename)] = txtfile.read()

        # ensure the parent directory exists
        output_file.parent.mkdir(exist_ok=True)

        with output_file.open("w") as csvfile:
            csvwriter = csv.DictWriter(csvfile, csv_fieldnames)
            for pagenum in sorted(txt_data.keys()):
                csvwriter.writerow({"page_number": pagenum, "text": txt_data[pagenum]})

        # save out text content as json
        # with output_file.open("w") as outfile:
        #     json.dump(txt_data, outfile)

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
        print(f"Error: input directory {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    # create output dir if it doesn't exist
    if not args.output_dir.is_dir():
        args.output_dir.mkdir()
        print(f"Error: creating output directory {args.output_dir}")

    collate_txt(args.input_dir, args.output_dir, show_progress=args.progress)


if __name__ == "__main__":
    main()
