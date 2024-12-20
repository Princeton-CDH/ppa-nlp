"""
Create a passim-friendly version of the input corpus. In addition to specifying
the initial input corpus and output corpus files (JSONL), a name for this corpus
of texts must be provided. This name is will be used by Passim for determining
which texts should be compared with one another (e.g., PPA texts vs. reference
poetry texts).

Example command line usage:
```
python create_passim_corpus.py poetry.jsonl poetry-passim.jsonl ref_poetry
```
"""

import argparse
import pathlib
import sys
from typing import Iterator

import orjsonl
from tqdm import tqdm


def transform_record(
    record: dict,
    corpus_name: str,
    id_field: str = "id",
    preserve_fields: bool = False,
):
    """
    Converts a record (dict) to a passim-friendly form.

    This method returns a new record (dict) with the following fields:
        * "id": record[id_field]
        * "corpus": corpus_name
        * "text": record["text"]

    Optionally, the output can preserve all fields found in the input records by
    setting preserve_fields to True.
    """
    # Validate input record
    if id_field not in record:
        raise ValueError(f"Record missing '{id_field}' field")

    out_record = {}
    # Copy over all fields
    if preserve_fields:
        if id_field != "id" and "id" in record:
            raise ValueError(f"Record has existing 'id' field")
        out_record |= record
    # Add required output
    out_record["id"] = record[id_field]
    out_record["corpus"] = corpus_name
    # If text field is missing, treat as blank page
    # TODO: Revisit whether to preserve blank pages. It should not meaningfully
    #       impact performance, but ensures a 1-1 line corresponds between the
    #       input and output JSONL files.
    out_record["text"] = record.get("text", "")
    return out_record


def build_passim_corpus(
    input_corpus: pathlib.Path,
    corpus_name: str,
    id_field: str = "id",
    preserve_fields: bool = False,
    disable_progress: bool = False,
) -> Iterator[dict]:
    """
    Converts an input text corpus (orjsonl-supported format) to a form suitable
    for passim as a record generator.

    The output records have the following form:
        * "id" field set to to the corresponding input record's id_field field
        * "corpus" field set to corpus_name
        * "text" field set to to the corresponding input record's "text" field

    Optionally, the output can preserve all fields found in the input records by
    setting preserve_fields to True.
    """
    record_progress = tqdm(
        orjsonl.stream(input_corpus),
        desc="Transforming records",
        disable=disable_progress,
    )
    for record in record_progress:
        yield transform_record(
            record,
            corpus_name,
            id_field,
            preserve_fields=preserve_fields,
        )


def save_passim_corpus(
    input_corpus: pathlib.Path,
    output_corpus: pathlib.Path,
    corpus_name: str,
    id_field: str,
    preserve_fields: bool = True,
    disable_progress: bool = False,
) -> None:
    """
    Converts an input text corpus (orjsonl-supported format) to a form suitable
    for passim and writes it to the provide output file path.
    """
    orjsonl.save(
        output_corpus,
        build_passim_corpus(
            input_corpus, corpus_name, id_field, preserve_fields, disable_progress
        ),
    )


def main():
    """
    Command-line access to creating passim-friendly version of a corpus
    """

    parser = argparse.ArgumentParser(
        description="Creates passim-friendly version of input corpus."
    )

    # Required arguments
    parser.add_argument(
        "input",
        help="Input text corpus; must be a JSONL file (compressed or not)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Filename where the resulting corpus (JSONL) should be saved.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "corpus_name",
        help="Name of corpus (to be used for new 'corpus' field)",
    )

    # Optional arguments
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--id-field",
        help="Specify name of field to be used as new 'id' field for the text record",
        default="id",
    )
    parser.add_argument(
        "--preserve-fields",
        help="Maintain all existing fields.",
        action="store_true",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input.is_file():
        print(f"Error: input corpus {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    if not args.output.parent.is_dir():
        print(
            f"Error: output corpus directory {args.output.parent} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.output.is_file():
        print(f"Error: output corpus {args.output} exists", file=sys.stderr)
        sys.exit(1)

    try:
        save_passim_corpus(
            args.input,
            args.output,
            args.corpus_name,
            id_field=args.id_field,
            preserve_fields=args.preserve_fields,
            disable_progress=not args.progress,
        )
    except Exception:
        print(f"Warning: Error encountered, deleting output...", file=sys.stderr)
        args.output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
