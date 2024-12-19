"""
Gather passim results (i.e., identified matches) at the page-level from
passim output files.
"""

import argparse
import pathlib
import sys

import orjsonl
from tqdm import tqdm


def get_span_annotation(alignment_record, include_excerpts=False):
    """
    Extract and rename fields from passim alignment record into a new
    page-level record.
    """
    span_annotation = {
        "ref_id": alignment_record["id"],
        "ref_corpus": alignment_record["corpus"],
        "ref_start": alignment_record["begin"],
        "ref_end": alignment_record["end"],
        "page_id": alignment_record["id2"],
        "page_start": alignment_record["begin2"],
        "page_end": alignment_record["end2"],
    }
    if include_excerpts:
        # Note: These excerpts are "aligned" versions with "-" characters
        # indicating where an insertion took place in the alignment algorithm
        span_annotation["aligned_page_excerpt"] = alignment_record["s2"]
        span_annotation["aligned_ref_excerpt"] = alignment_record["s1"]
    return span_annotation


def extract_passim_matches(passim_dir, include_excerpts=False, disable_progress=False):
    # Get passage-level matches
    align_dir = passim_dir.joinpath("align.json")
    if not align_dir.is_dir():
        raise ValueError("Error: Alignment directory '{align.json}' does not exist.")
    for filepath in align_dir.glob("*.json"):
        record_progress = tqdm(
            orjsonl.stream(filepath),
            desc=f"Extracting matches from {filepath.name}",
            disable=disable_progress,
        )
        for record in record_progress:
            yield get_span_annotation(record, include_excerpts)


def add_original_excerpts(
    page_records, input_corpus, ref_corpora=None, disable_progress=False
):
    """
    Add original excerpts to the page-level records.
    """
    if ref_corpora:
        # For tracking page reuse by reference text (i.e. poem)
        refs_to_pages = {}

    # Add page excerpts to the span annotations for each page record
    record_progress = tqdm(
        orjsonl.stream(input_corpus),
        total=len(page_records),
        desc="Adding page excerpts",
        disable=disable_progress,
    )
    for record in record_progress:
        page_id = record["id"]
        poem_spans = page_records[page_id]["poem_spans"]
        # Add (original) page excerpt to each poem span
        for span in poem_spans:
            start, end = span["page_start"], span["page_end"]
            span["page_excerpt"] = record["text"][start:end]
            if ref_corpora:
                # Add the page_id to the referenced text (i.e. poem)
                corpus_id = span["ref_corpus"]
                if corpus_id not in refs_to_pages:
                    refs_to_pages[corpus_id] = {}
                ref_id = span["ref_id"]
                if ref_id not in refs_to_pages[corpus_id]:
                    refs_to_pages[corpus_id][ref_id] = set()
                refs_to_pages[corpus_id][ref_id].add(page_id)

    # Optionally, add reference excerpts for provided reference corpora
    if ref_corpora:
        ref_corpus_set = set(ref_corpora)
        corpus_progress = tqdm(
            ref_corpus_set,
            desc="Reviewing reference corpora",
            disable=disable_progress or len(ref_corpus_set) == 1,
        )
        for ref_corpus in corpus_progress:
            record_progress = tqdm(
                orjsonl.stream(ref_corpus),
                desc="Adding reference excerpts",
                disable=disable_progress,
            )
            for ref_record in record_progress:
                # Skip corpus if it's never referenced
                ref_corpus = ref_record["corpus"]
                if ref_corpus not in refs_to_pages:
                    print(f"Warning: {ref_corpus} is never reused", file=sys.stderr)
                    break
                # Skip unreferenced texts
                ref_id = ref_record["id"]
                if ref_id not in refs_to_pages[ref_corpus]:
                    continue
                for page_id in refs_to_pages[ref_corpus][ref_id]:
                    # Update corresponding poem spans
                    for span in page_records[page_id]["poem_spans"]:
                        if (
                            span["ref_corpus"] == ref_corpus
                            and span["ref_id"] == ref_id
                        ):
                            start, end = span["ref_start"], span["ref_end"]
                            span["ref_excerpt"] = ref_record["text"][start:end]


def build_passim_output(
    input_corpus,
    passim_dir,
    output_path,
    disable_progress=False,
    include_excerpts=False,
    ref_corpora=None,
):
    # Get input ids
    input_ids = {record["id"] for record in orjsonl.stream(input_corpus)}

    # Initialize page-level annotation records
    page_records = {}
    for page_id in input_ids:
        page_records[page_id] = {"page_id": page_id, "n_spans": 0, "poem_spans": []}

    # Add passage-level matches to page-level records
    for match in extract_passim_matches(
        passim_dir, include_excerpts=include_excerpts, disable_progress=disable_progress
    ):
        page_id = match.pop("page_id")
        page_records[page_id]["poem_spans"].append(match)
        page_records[page_id]["n_spans"] += 1

    # Optionally, add excerpts to page-level annotation records
    if include_excerpts:
        add_original_excerpts(
            page_records,
            input_corpus,
            ref_corpora=ref_corpora,
            disable_progress=disable_progress,
        )

    # Write output to file
    orjsonl.save(output_path, page_records.values())


def main():
    """
    Command-line access to build a JSONL file gathering the page-level passim results.
    """
    parser = argparse.ArgumentParser(description="Build page-level passim results.")

    # Required arguments
    parser.add_argument(
        "input_corpus",
        help="The input corpus used in the passim run (ppa corpus)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "passim_output_dir",
        help="The top-level output directory for a passim run",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Filename for page-level passim output (JSON)",
        type=pathlib.Path,
    )
    # Optional arguments
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-excerpts",
        help="Include text excerpts (original & aligned) for each match",
        action="store_true",
    )
    parser.add_argument(
        "--ref-corpus",
        help="Reference corpus used in the passim run. Can specify multiple.",
        action="append",
        type=pathlib.Path,
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input_corpus.is_file():
        print(
            f"Error: input corpus {args.input_corpus} does not exist", file=sys.stderr
        )
        sys.exit(1)
    if not args.passim_output_dir.is_dir():
        print(
            f"Error: passim output directory {args.passim_output_dir} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.output.is_file():
        print(f"Error: output file {args.output} exists", file=sys.stderr)
        sys.exit(1)

    build_passim_output(
        args.input_corpus,
        args.passim_output_dir,
        args.output,
        not args.progress,
        args.include_excerpts,
        args.ref_corpus,
    )


if __name__ == "__main__":
    main()
