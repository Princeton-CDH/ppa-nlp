"""
Collect tag-attribute counts for the Chadwyck Healey corpus and output
them to a .jsonl file
"""

import argparse
import csv
import pathlib
import re
import sys

import bs4
import ftfy
from build_poem_corpus import get_poem_subdir
from tqdm import tqdm


def poem_path_generator(in_dir, poem_id_set=None):
    if poem_id_set is None:
        # Gather all poem records
        for poem_path in in_dir.rglob("*.tml"):
            # Skip nameless file
            if poem_path.name == ".tml":
                continue
            yield poem_path
    else:
        # Gather selected poem records
        for poem_id in poem_id_set:
            poem_dir = in_dir.joinpath(get_poem_subdir(poem_id))
            poem_path = poem_dir.joinpath(f"{poem_id}.tml")
            if not poem_path.is_file():
                print(f"Warning: poem file {poem_path} does not exist")
                continue
            yield poem_path


def get_tag_attr_values(tag_name, tag_attr, soup):
    attr_vals = set()
    for tag in soup.find_all(tag_name):
        attr_val = tag[tag_attr] if tag.has_attr(tag_attr) else "_"
        attr_vals.add(attr_val)
    return attr_vals


def get_poem_tag_stats(poem_path, tag_attr_pairs):
    with open(poem_path, encoding="latin1") as reader:
        raw_text = reader.read()
    # Get body block
    soup = bs4.BeautifulSoup(raw_text, "lxml", multi_valued_attributes=None).body
    poem_stats = {}
    for tag, attr in tag_attr_pairs:
        if tag not in poem_stats:
            poem_stats[tag] = {}
        poem_stats[tag][attr] = get_tag_attr_values(tag, attr, soup)
    return poem_stats


def get_corpus_tag_stats(in_dir, poem_ids=None, show_progress=True):
    # Set up progress bar
    disable_progress = not show_progress
    desc = "Reading poems"
    if poem_ids is None:
        # Gather all poem records
        bar_format = "{desc}: {n:,} poems read | elapsed: {elapsed}, {rate_fmt}"
        poem_progress = tqdm(
            poem_path_generator(in_dir),
            desc=desc,
            bar_format=bar_format,
            disable=disable_progress,
        )
    else:
        # Gather selected poems
        poem_id_set = set(poem_ids)
        n_poems = len(poem_id_set)
        poem_progress = tqdm(
            poem_path_generator(in_dir, poem_id_set=poem_id_set),
            desc=desc,
            total=n_poems,
            disable=disable_progress,
        )

    # TODO: Turn this into an input arg
    tag_attr_pairs = [("div", "type"), ("span", "class")]
    # Create corpus-level record
    corpus_stats = {}
    for tag, attr in tag_attr_pairs:
        if tag not in corpus_stats:
            corpus_stats[tag] = {}
        corpus_stats[tag][attr] = {}

    # Collect tag stats from each poem
    for poem_path in poem_progress:
        poem_id = poem_path.stem
        poem_stats = get_poem_tag_stats(poem_path, tag_attr_pairs)
        for tag, attr in tag_attr_pairs:
            for attr_val in poem_stats[tag][attr]:
                if attr_val not in corpus_stats[tag][attr]:
                    corpus_stats[tag][attr][attr_val] = set()
                corpus_stats[tag][attr][attr_val].add(poem_id)

    # Create records
    for tag, attr in tag_attr_pairs:
        for attr_val, poem_ids in corpus_stats[tag][attr].items():
            yield {"tag": tag, "attr": attr, "attr_val": attr_val, "poems": poem_ids}


def save_tag_stats(in_dir, out_fn, poem_ids=None, show_progress=True):
    field_names = ["tag", "attr", "attr_val", "poems"]
    with open(out_fn, newline="", mode="w") as file_handler:
        writer = csv.DictWriter(file_handler, field_names, dialect="excel-tab")
        writer.writeheader()
        for output in get_corpus_tag_stats(
            in_dir, poem_ids=poem_ids, show_progress=show_progress
        ):
            row = {
                "tag": output["tag"],
                "attr": output["attr"],
                "attr_val": output["attr_val"],
                "poems": ", ".join(output["poems"]),
            }
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Build Chadwyck Healey (sub)corpus.")
    # Required
    parser.add_argument(
        "input",
        help="Top-level poem directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Filename where the ouput should be saved",
        type=pathlib.Path,
    )
    # Optional
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--poems",
        help="Create subcorpus limited to the specified poems (by id)."
        "Can be repeated/",
        nargs="*",
        action="extend",
    )

    args = parser.parse_args()

    # Check that input directory exists
    if not args.input.is_dir():
        print(f"Error: input directory {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    # Add .jsonl extension to output if no extension is given
    output_fn = args.output
    if not output_fn.suffix:
        output_fn = args.output.with_suffix(".jsonl")
    # Check that output file does not exist
    if output_fn.is_file():
        print(f"Error: output file {args.output} already exists", file=sys.stderr)
        sys.exit(1)

    try:
        save_tag_stats(
            args.input, output_fn, poem_ids=args.poems, show_progress=args.progress
        )
    except Exception as err:
        # Remove output if it exists
        output_fn.unlink(missing_ok=True)
        # Rethrow error
        raise err


if __name__ == "__main__":
    main()
