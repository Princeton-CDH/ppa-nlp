"""
Utility script for PPA full-text dataset publication prep.

Takes a manually cleaned version of PPA metadata in CSV format, and
propagates refined author and pub_place fields to CSV and JSON metadata
in the specified ppa_corpus folder.
"""

import argparse
import json
import shutil
from pathlib import Path

import polars as pl


def refine_metadata(
    refined_metadata: Path, csv_metadata_path: Path, json_metadata_path: Path
) -> None:
    """Propagate refined author and publication-place values to corpus metadata."""
    refined_metadata_df = pl.read_csv(refined_metadata)
    # get the last update time of records in refined metadata (all records have added/updated time)
    refined_last_update = refined_metadata_df["updated"].max()
    df = pl.read_csv(csv_metadata_path)
    csv_columns = df.columns  # store columns original order for output file
    csv_total = df.height

    # join with refined metadata on work id so we can propagate work-specific
    # resolutions (author names are unambiguous but pub places are not)
    refined_df = df.join(
        refined_metadata_df.select("work_id", "author", "pub_place"),
        on="work_id",
        suffix="_refined",
        how="left",  # preserve all original records
    )
    # report how many instances of each field will be updated. A change is
    # counted only when the refined column carries a value that differs from
    # the original (the original is filled so comparisons with null are safe);
    # an empty refined value means "leave the original alone".
    author_changes_df = refined_df.filter(
        pl.col.author_refined.is_not_null()
        & pl.col.author_refined.ne(pl.col.author.fill_null(""))
    )
    num_author_changes = author_changes_df.height
    uniq_author_changes = (
        author_changes_df.select("author", "author_refined").unique().height
    )
    print(
        f"Updated author in {num_author_changes:,} works ({uniq_author_changes:,} unique replacements)"
    )

    pubplace_changes_df = refined_df.filter(
        pl.col.pub_place_refined.is_not_null()
        & pl.col.pub_place_refined.ne(pl.col.pub_place.fill_null(""))
    )
    num_pubplace_changes = pubplace_changes_df.height
    uniq_pubplace_changes = pubplace_changes_df.select(
        "pub_place", "pub_place_refined"
    ).n_unique()
    print(
        f"Updated pub_place in {num_pubplace_changes:,} works ({uniq_pubplace_changes:,} unique replacements)"
    )

    # then replace the original with the refined fields, but preserve the original
    # value whenever the refined set is absent
    refined_df = refined_df.with_columns(
        author=pl.coalesce("author_refined", "author"),
        pub_place=pl.coalesce("pub_place_refined", "pub_place"),
    ).drop("author_refined", "pub_place_refined")

    # identify any records added after the manual cleanup, and update author/pub_place
    # if there is an unambiguous mapping from the refined set
    post_refine_records = df.filter(pl.col.added.gt(refined_last_update))
    if post_refine_records.height:
        author_lookup = author_changes_df.select("author", "author_refined").unique()
        # omit any ambiguous authors (a single original author resolving to more
        # than one refined value cannot be applied automatically, and would make
        # the join below emit duplicate rows per work_id)
        author_lookup = author_lookup.filter(~author_lookup["author"].is_duplicated())

        pub_place_lookup = pubplace_changes_df.select(
            "pub_place", "pub_place_refined"
        ).unique()
        # omit any ambiguous placenames (i.e. Cambridge and Rochester could either be US or UK)
        pub_place_lookup = pub_place_lookup.filter(
            ~pub_place_lookup["pub_place"].is_duplicated()
        )
        # join and then filter to any that have changes
        post_refine_records = (
            post_refine_records.select("work_id", "author", "pub_place")
            .join(author_lookup, on="author", how="left")
            .join(pub_place_lookup, on="pub_place", how="left")
            .filter(
                # limit to records with at least one refined value
                ~(pl.col.author_refined.is_null() & pl.col.pub_place_refined.is_null())
            )
        )

        # update the refined data with these changes
        refined_df = (
            refined_df.join(
                post_refine_records.select(
                    "work_id", "author_refined", "pub_place_refined"
                ),
                on="work_id",
                how="left",
            )
            .with_columns(
                # take the refined value if not null, otherwise the previous value
                author=pl.when(pl.col.author_refined.is_not_null())
                .then(pl.col.author_refined)
                .otherwise(pl.col.author),
                pub_place=pl.when(pl.col.pub_place_refined.is_not_null())
                .then(pl.col.pub_place_refined)
                .otherwise(pl.col.pub_place),
            )
            .drop("author_refined", "pub_place_refined")
        )
    # clean whitespace for all author, pub_place, and publishers, regardless of refinement,
    # since unrefined values may have trailing whitespace
    refined_df = refined_df.with_columns(
        author=pl.col.author.str.strip_chars(),
        pub_place=pl.col.pub_place.str.strip_chars(),
        publisher=pl.col.publisher.str.strip_chars(),
    )

    # replace refined fields with unrefined and save the file
    assert refined_df.height == csv_total  # no records lost

    # Path.copy not available in 3.12, so use shutil to make a backup
    shutil.copy(csv_metadata_path, csv_metadata_path.with_suffix(".csv.bk"))
    # then replace the original with the refined version
    refined_df.select(csv_columns).write_csv(csv_metadata_path)

    # load json, update based on csv, write out
    # NOTE: the CSV and JSON are known to be equivalent and in the same order
    # (both are generated from the same data), so rows are paired in sequence
    # the per-row assert is a check to catch any mismatches
    with json_metadata_path.open() as jsonfile:
        json_metadata = json.load(jsonfile)
    refined_json_metadata = []
    for csv_row, json_row in zip(refined_df.iter_rows(named=True), json_metadata):
        assert csv_row["work_id"] == json_row["work_id"]
        if "author" in json_row:
            json_row["author"] = csv_row["author"]
        if "pub_place" in json_row:
            json_row["pub_place"] = csv_row["pub_place"]
        refined_json_metadata.append(json_row)
    # no rows lost
    assert len(json_metadata) == len(refined_json_metadata) == csv_total

    # make a backup and output the refined version
    shutil.copy(json_metadata_path, json_metadata_path.with_suffix(".json.bk"))
    with json_metadata_path.open("w") as jsonfile:
        json.dump(refined_json_metadata, jsonfile, indent=2)

    print(
        f"Updated metadata has been saved to {csv_metadata_path} and {json_metadata_path}"
    )


def main():
    """Parse command-line arguments and refine a PPA corpus metadata directory."""
    parser = argparse.ArgumentParser(
        description="Propagate PPA metadata standardization to updated metadata files"
    )
    parser.add_argument(
        "refined_metadata",
        help="PPA metadata file with standardized author/pub_place fields to propagate",
        type=Path,
    )

    parser.add_argument(
        "corpus_dir",
        help="PPA full-text corpus directory; expected to contain "
        "work-level metadata files ppa_metadata.csv and ppa_metadata.json",
        type=Path,
    )

    args = parser.parse_args()
    if not args.refined_metadata.is_file():
        raise SystemExit(f"Input metadata file {args.refined_metadata} does not exist")

    if not args.corpus_dir.is_dir():
        raise SystemExit(f"PPA corpus path {args.corpus_dir} is not a directory")

    # we expect two formats
    csv_metadata_path = args.corpus_dir / "ppa_metadata.csv"
    json_metadata_path = args.corpus_dir / "ppa_metadata.json"
    if not csv_metadata_path.exists():
        raise SystemExit(f"PPA metadata file {csv_metadata_path} not found")
    if not json_metadata_path.exists():
        raise SystemExit(f"PPA metadata file {json_metadata_path} not found")

    refine_metadata(args.refined_metadata, csv_metadata_path, json_metadata_path)


if __name__ == "__main__":
    main()
