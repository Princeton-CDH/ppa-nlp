import argparse
import sys
from pathlib import Path

import orjsonl
from path_utils import get_image_relpath
from tqdm import tqdm


def add_image_paths(in_jsonl, ext=None, show_progress=True):
    progress_bar = tqdm(
        orjsonl.stream(in_jsonl),
        desc="Adding page paths",
        bar_format="{desc}: processed {n:,} pages{postfix} | elapsed: {elapsed}",
        disable=not show_progress,
    )
    for page in progress_bar:
        work_id = page["work_id"]
        page_num = page["order"]
        image_relpath = get_image_relpath(work_id, page_num)
        if ext is not None:
            image_relpath = image_relpath.with_suffix(ext)
        # Add relative path to record
        page["image_path"] = str(image_relpath)
        yield page


def save_corpus_with_image_relpaths(in_jsonl, out_jsonl, ext=None, show_progress=True):
    orjsonl.save(out_jsonl, add_image_paths(in_jsonl, ext=ext, show_progress=True))


def main():
    parser = argparse.ArgumentParser(
        description="Add image (relative) paths to PPA full-text corpus",
    )
    # Required arguments
    parser.add_argument(
        "input",
        help="PPA full-text corpus to add page-level image paths to; "
        "must be a JSONL file (compresed or not)",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Filename where output corpus should be saved",
        type=Path,
    )
    # Optional argument
    parser.add_argument(
        "--ext",
        help="Extension to use for all image paths instead of the source-level defaults",
    )
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    # If output filename does not have an extension, add jsonl
    out_jsonl = args.output
    if out_jsonl.suffix == "":
        out_jsonl = out_jsonl.with_suffix(".jsonl")

    # Validate arguments
    if not args.input.is_file():
        print(f"Input {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    if out_jsonl.is_file():
        print(f"Output {args.output} already exist", file=sys.stderr)
        sys.exit(1)
    if args.ext:
        if args.ext[0] != ".":
            print(f"Extension must start with '.'", file=sys.stderr)
            sys.exit(1)

    save_corpus_with_image_relpaths(
        args.input,
        out_jsonl,
        ext=args.ext,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
