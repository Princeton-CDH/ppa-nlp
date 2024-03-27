import argparse

import orjsonl
from tqdm import tqdm


def filter_pages(input_filename, source_ids):
    """Takes a filename for a PPA full-text corpus in a format orjsonl supports
    and a list of source ids. Returns a generator of filtered pages from the
    full corpus corresponding to the list of ids.
    """
    # convert list of source ids to set for fast hashmap lookup
    source_ids = set(source_ids)
    selected_pages = 0
    # TODO: handle errors orjson.JSONDecodeError; FileNotFoundError
    progress_pages = tqdm(
        orjsonl.stream(input_filename),
        desc="Filtering",
        bar_format="{desc}: {n:,} pages{postfix} | elapsed: {elapsed}",
    )
    for page in progress_pages:
        # page data does not include source id, but does include work id
        # which is either source id (for full works) or
        # source id plus first page number (for articles/excerpts)
        if page["work_id"].split("-p")[0] in source_ids:
            # keep track of how many have been selected for reporting in
            # progress bar
            selected_pages += 1
            progress_pages.set_postfix_str(f"selected {selected_pages:,}")
            yield page

    # NOTE: other filters could be implemented here later, e.g.
    # based on HathiTrust page tags like UNTYPICAL_PAGE


def save_filtered_corpus(input_filename, output_filename, source_ids):
    orjsonl.save(output_filename, filter_pages(input_filename, source_ids))


def main():
    # command-line access to filtering the corpus
    parser = argparse.ArgumentParser(
        description="Filters PPA full-text corpus by list of source ids",
    )
    parser.add_argument(
        "input",
        help="path to PPA full-text corpus to be "
        + "filtered; must be a JSONL file (compressed or not)",
    )
    parser.add_argument(
        "output", help="filename where the filtered corpus should be saved"
    )
    parser.add_argument("idfile", help="filename with list of source ids, one per line")

    args = parser.parse_args()
    with open(args.idfile) as idfile:
        source_ids = [line.strip() for line in idfile.readlines()]
    save_filtered_corpus(args.input, args.output, source_ids)


if __name__ == "__main__":
    main()
