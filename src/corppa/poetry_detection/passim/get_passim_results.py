# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

"""
Build passim page-level and span-level matches from passim output files.

Examples:
    get_passim_results.py --ppa-passim-corpus ppa_passim.jsonl --ref-corpus ref.jsonl \
        --passim-dir passim_output --page-results passim_page_results.jsonl \
        --span-results passim_spans.csv
    get_passim_results.py --ppa-passim-corpus ppa_passim.jsonl --ref-corpus ref_a.jsonl \
        --ref-corpus ref_b.jsonl --passim-dir passim_output \
        --page-results passim_page_results.jsonl --span-results passim_spans.csv \
        --ppa-text-corpus ppa.jsonl.gz
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

import orjsonl
from tqdm import tqdm

from corppa.poetry_detection.core import LabeledExcerpt


def get_page_texts(page_ids: Iterable[str], text_corpus: Path) -> dict[str, None | str]:
    """
    Gathers the texts from corpus file for the specified pages (by id), returns
    a dictionary mapping page ids to page texts.
    """
    page_texts = {}
    for page in orjsonl.stream(text_corpus):
        page_id = page["id"]
        if page_id in page_ids:
            page_texts[page_id] = page.get("text", "")
    return page_texts


def build_passim_excerpt(
    page_id: str, span_record: dict[str, Any], ppa_page_text: None | str = None
) -> LabeledExcerpt:
    """
    Creates passim excerpt using the passed in page id and passim span record as
    produced by `build_passim_page_results.`

    Optionally, can provide the original PPA page text to "correct" the excerpt
    for any text transformations applied during the passim pipeline.
    """
    excerpt = LabeledExcerpt(
        page_id=page_id,
        ppa_span_start=span_record["page_start"],
        ppa_span_end=span_record["page_end"],
        ppa_span_text=span_record["ppa_excerpt"],
        detection_methods={"passim"},
        poem_id=span_record["poem_id"],
        ref_corpus=span_record["ref_corpus"],
        ref_span_start=span_record["ref_start"],
        ref_span_end=span_record["ref_end"],
        ref_span_text=span_record["ref_excerpt"],
        identification_methods={"passim"},
        notes=f"passim: {span_record['matches']} char matches",
    )
    if ppa_page_text:
        # Correct excerpt if we have the original page text
        excerpt = excerpt.correct_page_excerpt(ppa_page_text)
    return excerpt


def get_passim_span(alignment_record) -> dict[str, Any]:
    """
    Extract span record from and rename fields from passim alignment record into a new
    page-level record.
    """
    span_record = {
        "poem_id": alignment_record["id"],
        "ref_corpus": alignment_record["corpus"],
        "ref_start": alignment_record["begin"],
        "ref_end": alignment_record["end"],
        "page_id": alignment_record["id2"],
        "page_start": alignment_record["begin2"],
        "page_end": alignment_record["end2"],
        "matches": alignment_record["matches"],
        # Note: "aligned" excerpts use "-" to indicate insertions
        "aligned_ref_excerpt": re.sub(r"\s", " ", alignment_record["s1"]),
        "aligned_ppa_excerpt": re.sub(r"\s", " ", alignment_record["s2"]),
    }
    return span_record


def extract_passim_spans(
    passim_dir: Path,
    disable_progress: bool = False,
) -> Generator[dict[str, Any]]:
    """
    Extracts all span-level matches identified by passim returned as a generator
    """
    align_dir = passim_dir.joinpath("align.json")
    if not align_dir.is_dir():
        raise ValueError(f"Error: Alignment directory '{align_dir}' does not exist")
    for filepath in align_dir.glob("*.json"):
        record_progress = tqdm(
            orjsonl.stream(filepath),
            desc=f"Extracting matches from {filepath.name}",
            disable=disable_progress,
        )
        for record in record_progress:
            yield get_passim_span(record)


def add_excerpts(
    page_results: dict[str, dict[str, Any]],
    ppa_passim_corpus: Path,
    ref_corpora: Iterable[Path],
    disable_progress: bool = False,
) -> None:
    """
    Add original PPA and reference excerpts to the span within page results
    """
    # For tracking page reuse by reference text
    refs_to_pages: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    # Add PPA excerpts
    ppa_progress = tqdm(
        orjsonl.stream(ppa_passim_corpus),
        total=len(page_results),
        desc="Adding PPA excerpts",
        disable=disable_progress,
    )
    for ppa_record in ppa_progress:
        page_id = ppa_record["id"]
        poem_spans = page_results[page_id]["poem_spans"]
        # Add PPA excerpt to each poem span
        for span in poem_spans:
            start, end = span["page_start"], span["page_end"]
            span["ppa_excerpt"] = ppa_record["text"][start:end]
            # Add the page_id to the referenced text
            corpus_id = span["ref_corpus"]
            poem_id = span["poem_id"]
            refs_to_pages[corpus_id][poem_id].add(page_id)

    # Add reference excerpts
    for ref_corpus in ref_corpora:
        ref_progress = tqdm(
            orjsonl.stream(ref_corpus),
            desc=f"Adding {ref_corpus} reference excerpts",
            disable=disable_progress,
        )
        for ref_record in ref_progress:
            corpus_id = ref_record["corpus"]
            poem_id = ref_record["id"]
            if poem_id not in refs_to_pages[corpus_id]:
                # Skip unreferenced texts
                continue
            for page_id in refs_to_pages[corpus_id][poem_id]:
                # Add reference excerpts to corresponding spans
                for span in page_results[page_id]["poem_spans"]:
                    if span["ref_corpus"] == corpus_id and span["poem_id"] == poem_id:
                        start, end = span["ref_start"], span["ref_end"]
                        span["ref_excerpt"] = ref_record["text"][start:end]


def build_passim_page_results(
    ppa_passim_corpus: Path,
    ref_corpora: Iterable[Path],
    passim_dir: Path,
    disable_progress: bool = False,
):
    # Initialize page-level results
    page_results: dict[str, dict[str, Any]] = {}
    page_progress = tqdm(
        orjsonl.stream(ppa_passim_corpus),
        desc="Initializing PPA page-level results",
        disable=disable_progress,
    )
    for record in page_progress:
        page_id = record["id"]
        page_results[page_id] = {"page_id": page_id, "n_spans": 0, "poem_spans": []}

    # Add passage-level matches to page-level records
    for match in extract_passim_spans(passim_dir, disable_progress=disable_progress):
        page_id = match.pop("page_id")
        page_results[page_id]["poem_spans"].append(match)
        page_results[page_id]["n_spans"] += 1

    # Add excerpts to span records
    add_excerpts(
        page_results, ppa_passim_corpus, ref_corpora, disable_progress=disable_progress
    )
    return page_results


def write_passim_results(
    ppa_passim_corpus: Path,
    ref_corpora: Iterable[Path],
    passim_dir: Path,
    out_page_results: Path,
    out_span_results: Path,
    ppa_text_corpus: None | Path = None,
    disable_progress: bool = False,
) -> None:
    # Get page-level results
    page_results = build_passim_page_results(
        ppa_passim_corpus, ref_corpora, passim_dir, disable_progress=disable_progress
    )
    # Optionally, gather relevant original PPA page texts
    ppa_page_texts = {}
    if ppa_text_corpus:
        ppa_page_texts = get_page_texts(page_results.keys(), ppa_text_corpus)

    # Write page-level & span-level output by page
    page_progress = tqdm(
        page_results.items(),
        desc="Writing results",
        disable=disable_progress,
    )

    with open(out_span_results, mode="w", newline="") as csvfile:
        # Passim-specific fields
        passim_fields = ["matches", "aligned_ref_excerpt", "aligned_ppa_excerpt"]
        # Add additional passim-specific fields
        fieldnames = LabeledExcerpt.fieldnames() + passim_fields
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for page_id, record in page_progress:
            # Write page-level results to file
            orjsonl.append(out_page_results, record)

            # Write span-level results to file
            page_text = ppa_page_texts.get(page_id)
            for span_record in record["poem_spans"]:
                excerpt = build_passim_excerpt(
                    page_id, span_record, ppa_page_text=page_text
                )
                row_fields = excerpt.to_csv()
                row_fields.update({key: span_record[key] for key in passim_fields})
                writer.writerow(row_fields)


def main():
    """
    Command-line access to build page-level and span-level passim results.
    """
    parser = argparse.ArgumentParser(description="Build passim results.")
    # Required arguments
    parser.add_argument(
        "--ppa-passim-corpus",
        help="Path to PPA passim-friendly corpus file (JSONL)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ref-corpus",
        help="Path to reference passim-friendly corpus file (JSONL). Can specify multiple",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--passim-dir",
        help="The top-level directory containing the ouput of the passim run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--page-results",
        help="Filename for the page-level passim results (JSONL)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--span-results",
        help="Filename for the span-level passim results (CSV)",
        type=Path,
        required=True,
    )

    # Optional arguments
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ppa-text-corpus",
        help="Original PPA text corpus file (JSONL) for correcting identified excerpts",
        type=Path,
    )

    args = parser.parse_args()

    # Validate input/output paths
    if not args.ppa_passim_corpus.is_file():
        print(
            f"Error: PPA passim corpus {args.ppa_passim_corpus} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)
    for ref in args.ref_corpus:
        if not ref.is_file():
            print(
                f"Error: reference corpus {ref} does not exist",
                file=sys.stderr,
            )
            sys.exit(1)
    if not args.passim_dir.is_dir():
        print(
            f"Error: Passim directory {args.passim_dir} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.page_results.is_file():
        print(
            f"Error: Output page-level results file {args.page_results} exists",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.span_results.is_file():
        print(
            f"Error: Output span-level results file {args.span_results} exists",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.ppa_text_corpus and not args.ppa_text_corpus.is_file():
        print(
            f"Error: ppa text corpus {args.ppa_text_corpus} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)

    write_passim_results(
        args.ppa_passim_corpus,
        args.ref_corpus,
        args.passim_dir,
        args.page_results,
        args.span_results,
        ppa_text_corpus=args.ppa_text_corpus,
        disable_progress=not args.progress,
    )


if __name__ == "__main__":
    main()
