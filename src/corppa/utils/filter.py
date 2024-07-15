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
corppa-filter-corpus path/to/ppa_pages.jsonl output/ppa_subset_pages.jsonl --idfile my_ids.txt
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
    input_filename: str,
    source_ids: list[str] | None = None,
    disable_progress: bool = False,
    include_filter: dict | None = None,
    exclude_filter: dict | None = None,
) -> Iterator[dict]:
    """Takes a filename for a PPA full-text corpus in a format orjsonl supports
    and one or more options for filtering that corpus. Returns a generator of
    filtered pages from the full corpus corresponding to the list of ids.
    At least one filtering option must be specified.
    Displays progress with :mod:`tqdm` progress bar unless disabled.

    :param input_filename: str, filename for corpus input
    :param source_ids: list of str, source ids to include in filtered pages
    :param include_filter: dict of key-value pairs for pages to include in
        the filtered page set; equality check against page data attributes (optional)
    :param exclude_filter: dict of key-value pairs for pages to exclude from
        the filtered page set; equality check against page data attributes (optional)
    :param disable_progress: boolean, disable progress bar (optional, default: False)
    :returns: generator of dict with page data
    :raises: FileNotFoundError, orjson.JSONDecodeError
    """
    # convert list of source ids to set for fast hashmap lookup
    if not any([source_ids, include_filter, exclude_filter]):
        raise ValueError(
            "At least one filter must be specified (source_ids, include_filter, exclude_filter)"
        )

    if source_ids is not None:
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

        # list of flags for inclusion/exclusion, for combining filters
        include_page = []

        # if list of source ids is specified, set true or false for inclusion
        if source_ids:
            include_page.append(page["work_id"].split("-p")[0] in source_ids)

        # if key-value pairs for inclusion are specified, filter
        if include_filter:
            # multiple include filters use OR logic:
            # if any include filter applies, this page should be included
            include_page.append(
                any(page[key] == val for key, val in include_filter.items())
            )

        # if key-value pairs for exclusion are specified, filter
        if exclude_filter:
            # if any exclusion filter applies, this page should not be included
            include_page.append(
                not any(page[key] == val for key, val in exclude_filter.items())
            )

        # make sure we have at least one True flag and all flags are True
        if include_page and all(include_page):
            # keep track of how many have been selected for reporting in
            # progress bar
            selected_pages += 1
            progress_pages.set_postfix_str(f"selected {selected_pages:,}")
            yield page


def save_filtered_corpus(
    input_filename: str,
    output_filename: str,
    idfile: str | None = None,
    disable_progress: bool = False,
    include_filter: dict | None = None,
    exclude_filter: dict | None = None,
) -> None:
    """Takes a filename for input PPA full-text corpus in a format
    orjsonl supports, filename where filtered corpus should be saved,
    and a filename with a list of source ids, one id per line.
    At least one filter must be specified.
    Calls :meth:`filter_pages`.

    :param input_filename: str, filename for corpus input
    :param output_filename: str, filename for filtered corpus output
    :param idfile: str, filename for list of source ids (optional)
    :param include_filter: dict of key-value pairs for pages to include in
        the filtered page set; equality check against page data attributes (optional)
    :param exclude_filter: dict of key-value pairs for pages to exclude from
        the filtered page set; equality check against page data attributes (optional)
    :param disable_progress: boolean, disable progress bar (optional, default: False)
    """

    source_ids = None

    # at least one filter is required
    if not any([idfile, include_filter, exclude_filter]):
        raise ValueError(
            "At least one filter must be specified (idfile, include_filter, exclude_filter)"
        )

    # if an id file is specifed, read and generate a list of ids to include
    if idfile:
        with open(idfile) as idfile_content:
            source_ids = [line.strip() for line in idfile_content]

    # use orjsonl to stream filtered pages to specified output file
    orjsonl.save(
        output_filename,
        filter_pages(
            input_filename,
            source_ids=source_ids,
            disable_progress=disable_progress,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
        ),
    )
    # zero size file means no pages were selected; cleanup
    # (report?)
    # if os.path.getsize(output_filename) == 0:
    #     os.remove(output_filename)


class MergeKeyValuePairs(argparse.Action):
    """
    custom argparse action to split a KEY=VALUE argument and append the pairs to a dictionary.
    """

    # adapted from https://stackoverflow.com/a/77148515/9706217

    # NOTE: we may want a multidict here to support multiple values for the same key

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
        description="Filters PPA full-text corpus by list of source ids",
    )
    parser.add_argument(
        "input",
        help="PPA full-text corpus to be "
        + "filtered; must be a JSONL file (compressed or not)",
    )
    parser.add_argument(
        "output", help="filename where the filtered corpus should be saved"
    )
    parser.add_argument(
        "--progress",
        help="Show progress",
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
        help="filename with list of source ids, one per line",
        required=False,
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
    if not any([args.idfile, args.include, args.exclude]):
        parser.error("At least one filter option must be specified")

    # TODO: use file or pathlib types?

    if args.idfile:
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
            args.input,
            output_filename,
            idfile=args.idfile,
            disable_progress=disable_progress,
            include_filter=args.include,
            exclude_filter=args.exclude,
        )
    except (FileNotFoundError, JSONDecodeError) as err:
        # catch known possible errors and display briefly
        # with the type of error and the brief message
        print(f"{err.__class__.__name__}: {err}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
