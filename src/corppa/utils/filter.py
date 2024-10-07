"""
Utility for filtering PPA full-text corpus to work with a subset of
pages.

Currently supports the following types of filtering:
  * List of PPA work ids (as a text file, id-per-line)
  * CSV file specifying work pages (by digital page number) (csv, page-per-line)
  * Filtering by key-value pair for either inclusion or exclusion

These filtering options can be combined, generally as a logical AND. Pages filtered
by work ids or page numbers will be further filtered by the key-value logic. In cases
where both work- and page-level filtering occurs, works not specified in the page
filtering are included in full. Works that are specified in both will be limited to the
pages specified in page-level filtering.

Filter methods can be run via command-line or python code. Filtering takes a jsonl file
(compressed or not) as input, and will produce a jsonl file (compressed or not) as output.
The input and output filenames can use any extension supported by any extension supported
by :mod:`orjsonl`, with or without compression; e.g. `.jsonl`, `.jsonl.gz`, `.jsonl.bz2`, etc.

Example command line usages:
```
corppa-filter-corpus path/to/ppa_pages.jsonl output/ppa_subset_pages.jsonl --idfile my_ids.txt
```

```
corppa-filter-corpus path/to/ppa_pages.jsonl output/ppa_subset_pages.jsonl --pg-file pages.csv --include key=value
```
"""

import argparse
import csv
import pathlib
import sys
from typing import Iterator

import orjsonl
from orjson import JSONDecodeError
from tqdm import tqdm


def filter_pages(
    input_filename: pathlib.Path,
    work_ids: list[str] | None = None,
    work_pages: dict | None = None,
    include_filter: dict | None = None,
    exclude_filter: dict | None = None,
    disable_progress: bool = False,
) -> Iterator[dict]:
    """Takes a filename for a PPA full-text corpus in a format orjsonl supports
    and one or more options for filtering that corpus. Returns a generator of
    filtered pages from the full corpus corresponding to the list of ids.
    At least one filtering option must be specified.
    Displays progress with :mod:`tqdm` progress bar unless disabled.

    :param input_filename: pathlib.Path, filename for corpus input
    :param work_ids: list of str, work ids to include in filtered pages (optional)
    :param work_pages: dict of str-set[int] pairs, specifies the set of digital page
        numbers of a work (by work id) to be filtered to be filtered to (optional)
    :param include_filter: dict of key-value pairs for pages to include in
        the filtered page set; equality check against page data attributes (optional)
    :param exclude_filter: dict of key-value pairs for pages to exclude from
        the filtered page set; equality check against page data attributes (optional)
    :param disable_progress: boolean, disable progress bar (optional, default: False)
    :returns: generator of dict with page data
    :raises: FileNotFoundError, orjson.JSONDecodeError
    """
    # at least one filter is required
    if not any([work_ids, work_pages, include_filter, exclude_filter]):
        raise ValueError(
            "At least one filter must be specified (work_ids, work_pages, include_filter, exclude_filter)"
        )

    if work_ids is not None:
        # convert list of work ids to set for fast hashmap lookup
        work_ids = set(work_ids)
    # if work pages is provided, update work ids set
    if work_pages is not None:
        if work_ids is None:
            work_ids = set(work_pages)
        else:
            work_ids |= set(work_pages)

    selected_pages = 0
    progress_pages = tqdm(
        orjsonl.stream(input_filename),
        desc="Filtering",
        bar_format="{desc}: checked {n:,} pages{postfix} | elapsed: {elapsed}",
        disable=disable_progress,
    )
    for page in progress_pages:
        # if work ids is specified and id does not match, skip
        if work_ids:
            if page["work_id"] not in work_ids:
                continue

        # if work pages is specified, filter
        if work_pages:
            # if work id is in indexed, skip pages not include in its set
            # NOTE: works specified in the work ids filter but not the work_pages
            #       filter will be included.
            if page["work_id"] in work_pages:
                if page["order"] not in work_pages[page["work_id"]]:
                    continue

        # if key-value pairs for inclusion are specified, filter
        if include_filter:
            # multiple include filters use OR logic:
            # if include filter does not apply, skip this page
            if not any(page[key] == val for key, val in include_filter.items()):
                continue

        # if key-value pairs for exclusion are specified, filter
        if exclude_filter:
            # if exclude filter matches, skip this page
            if any(page[key] == val for key, val in exclude_filter.items()):
                continue

        # keep track of how many have been selected for reporting in
        # progress bar
        selected_pages += 1
        progress_pages.set_postfix_str(f"selected {selected_pages:,}")
        yield page


def save_filtered_corpus(
    input_filename: pathlib.Path,
    output_filename: pathlib.Path,
    idfile: pathlib.Path | None = None,
    pgfile: pathlib.Path | None = None,
    include_filter: dict | None = None,
    exclude_filter: dict | None = None,
    disable_progress: bool = False,
) -> None:
    """Takes a filename for input PPA full-text corpus in a format
    orjsonl supports, filename where filtered corpus should be saved,
    and a filename with a list of work ids, one id per line.
    At least one filter must be specified.
    Calls :meth:`filter_pages`.

    :param input_filename: pathlib.Path, filepath for corpus input
    :param output_filename: pathlib.Path, filepath for filtered corpus output
    :param idfile: pathlib.Path, filepath for list of work ids (optional)
    :param pgfile: pathlib.Path, filepath for list of pages (optional)
    :param include_filter: dict of key-value pairs for pages to include in
        the filtered page set; equality check against page data attributes (optional)
    :param exclude_filter: dict of key-value pairs for pages to exclude from
        the filtered page set; equality check against page data attributes (optional)
    :param disable_progress: boolean, disable progress bar (optional, default: False)
    """

    work_ids = None
    work_pages = None

    # at least one filter is required
    if not any([idfile, pgfile, include_filter, exclude_filter]):
        raise ValueError(
            "At least one filter must be specified (idfile, pgfile, include_filter, exclude_filter)"
        )

    # if an id file is specified, read and generate a list of ids to include
    if idfile:
        with open(idfile) as idfile_content:
            work_ids = [line.strip() for line in idfile_content]

    # if a page file is specified, build page index (work id -> page set) from file
    if pgfile:
        work_pages = {}
        with open(pgfile, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            # Check header
            if (
                "work_id" not in reader.fieldnames
                or "page_num" not in reader.fieldnames
            ):
                raise ValueError(
                    f'pgfile {pgfile} must include fields "work_id" and "page_num"'
                )

            for row in reader:
                if row["work_id"] not in work_pages:
                    work_pages[row["work_id"]] = set()
                work_pages[row["work_id"]].add(int(row["page_num"]))

    # use orjsonl to stream filtered pages to specified output file
    orjsonl.save(
        output_filename,
        filter_pages(
            input_filename,
            work_ids=work_ids,
            work_pages=work_pages,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            disable_progress=disable_progress,
        ),
    )


class MergeKeyValuePairs(argparse.Action):
    """
    custom argparse action to split a KEY=VALUE argument and append the pairs to a dictionary.
    """

    # adapted from https://stackoverflow.com/a/77148515/9706217

    # NOTE: in future, we may want an option to store multiple values for
    # the same key, perhaps using multidict or dict of key -> set(possible values)

    def __call__(self, parser, args, values, option_string=None):
        previous = getattr(args, self.dest, None) or dict()
        try:
            added = dict(map(lambda x: x.split("="), values))
        except ValueError:
            raise argparse.ArgumentError(
                self, f'Could not parse argument "{values}" as k1=v1 k2=v2 ... format'
            )
        merged = {**previous, **added}
        setattr(args, self.dest, merged)


def main():
    """Command-line access to filtering the corpus. Available as
    `corppa-filter-corpus` when this package is installed with pip."""

    parser = argparse.ArgumentParser(
        description="Filters PPA full-text corpus",
    )
    parser.add_argument(
        "input",
        help="PPA full-text corpus to be "
        + "filtered; must be a JSONL file (compressed or not)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="filename where the filtered corpus should be saved",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cleanup",
        help="Remove empty output file if no pages are relected",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    filter_args = parser.add_argument_group(
        "filters",
        "Options for filtering pages. MUST include at least one. "
        + "When multiple filters are specified, they are all combined (AND). "
        + "If multiple include/exclude filters are specified, a page is "
        + "included/excluded if ANY key=value pairs match.",
    )
    filter_args.add_argument(
        "-i",
        "--idfile",
        help="File containing a list of work ids (one per line) to filter to",
        type=pathlib.Path,
        required=False,
    )
    filter_args.add_argument(
        "--pgfile",
        help="CSV file containing the list of pages to filter to. File must have a header "
        + 'with fields named "work_id" and "page_num".',
        type=pathlib.Path,
    )
    filter_args.add_argument(
        "--include",
        nargs="*",
        action=MergeKeyValuePairs,
        metavar="KEY=VALUE",
        help='Include pages by attribute: add key-value pairs as key=value or key="another value". '
        + "(no spaces around =, use quotes for values with spaces)",
    )
    filter_args.add_argument(
        "--exclude",
        nargs="*",
        action=MergeKeyValuePairs,
        metavar="KEY=VALUE",
        help='Exclude pages by attribute: add key-value pairs as key=value or key="another value". '
        + "(no spaces around =, use quotes for values with spaces)",
    )

    args = parser.parse_args()
    # progress bar is enabled by default; disable if requested
    disable_progress = not args.progress

    # at least one filter must be specified
    # check that one of idfile, include, or exclude is specified
    if not any([args.idfile, args.pgfile, args.include, args.exclude]):
        parser.error("At least one filter option must be specified")

    if args.idfile:
        if not args.idfile.is_file():
            print(f"Error: idfile {args.idfile} does not exist", file=sys.stderr)
            sys.exit(1)
        elif args.idfile.stat().st_size == 0:
            print(f"Error: idfile {args.idfile} is zero size", file=sys.stderr)
            sys.exit(1)

    if args.pgfile:
        if not args.pgfile.is_file():
            print(f"Error: pgfile {args.pgfile} does not exist", file=sys.stderr)
            sys.exit(1)
        elif args.pgfile.stat().st_size == 0:
            print(f"Error: pgfile {args.pgfile} is zero size", file=sys.stderr)
            sys.exit(1)

    # if requested output filename has no extension, add jsonl
    output_filepath = args.output
    if output_filepath.suffix == "":
        output_filepath = output_filepath.with_suffix(".jsonl")

    if output_filepath.is_file():
        print(
            f"Error: requested output file {args.output} already exists; not overwriting",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        save_filtered_corpus(
            args.input,
            output_filepath,
            idfile=args.idfile,
            pgfile=args.pgfile,
            include_filter=args.include,
            exclude_filter=args.exclude,
            disable_progress=disable_progress,
        )
    except (FileNotFoundError, JSONDecodeError) as err:
        # catch known possible errors and display briefly
        # with the type of error and the brief message
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)
        sys.exit(1)

    # check if output file exists but is zero size (i.e., no pages selected)
    if output_filepath.is_file() and output_filepath.stat().st_size == 0:
        # if cleanup is disabled, remove and report
        if args.cleanup:
            output_filepath.unlink()
            print(
                f"No pages were selected, removing empty output file {output_filepath}"
            )
        # otherwise just report
        else:
            print(f"No pages were selected, output file {output_filepath} is empty")


if __name__ == "__main__":
    main()
