#!/usr/bin/env python
import csv
import pathlib
import re
from glob import iglob
from itertools import batched

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# for convenience, assume the poetry reference data directory is
# available relative to wherever this script is called from
REF_DATA_DIR = pathlib.Path("poetry-reference-data")
TEXT_PARQUET_FILE = REF_DATA_DIR / "poems.parquet"
META_PARQUET_FILE = REF_DATA_DIR / "poem_metadata.parquet"
# csv files to supplement .txt files
POETRY_FOUNDATION_CSV = REF_DATA_DIR / "poetryfoundationdataset.csv"
CHADWYCK_HEALEY_CSV = REF_DATA_DIR / "chadwyck_healey_metadata.csv"
# define source ids to ensure we are consistent
SOURCE_ID = {
    "Poetry Foundation": "poetry-foundation",
    "Chadwyck-Healey": "chadwyck-healey",
}


def compile_text(data_dir, output_file):
    """Compile reference poems into a parquet file for quick identification
    of poetry excerpts based on matching text. Looks for text files in
    directories under `data_dir`; uses the filename stem as poem identifier
    and the containing directory name as the id for the source reference corpus.
    Also looks for and includes content from `poetryfoundationdataset.csv`
    contained in the data directory.
    """

    # parquet file schema:
    # - poem id
    # - text of the poem
    # - source (identifier for the reference corpus)
    schema = pa.schema(
        [("id", pa.string()), ("text", pa.string()), ("source", pa.string())]
    )
    # open a parquet writer so we can add records in chunks
    pqwriter = pq.ParquetWriter(output_file, schema)

    # handle files in batches
    # look for .txt files in nested directories; use parent directory name as
    # the reference corpus source name/id
    for chunk in batched(iglob(f"{data_dir}/**/*.txt"), 1000):
        chunk_files = [pathlib.Path(f) for f in chunk]
        ids = [f.stem for f in chunk_files]
        sources = [f.parent.name for f in chunk_files]
        texts = [f.open().read() for f in chunk_files]
        # create and write a record batch
        record_batch = pa.RecordBatch.from_arrays(
            [ids, texts, sources], names=["id", "text", "source"]
        )
        pqwriter.write_batch(record_batch)

    # poetry foundation text content is included in the csv file
    if POETRY_FOUNDATION_CSV.exists():
        # TODO: convert this to use polars
        with POETRY_FOUNDATION_CSV.open() as pf_csvfile:
            csv_reader = csv.DictReader(pf_csvfile)
            # process csv file in chunks
            for chunk in batched(csv_reader, 1000):
                ids = [row["Poetry Foundation ID"] for row in chunk]
                texts = [row["Content"] for row in chunk]
                source = [SOURCE_ID["Poetry Foundation"]] * len(ids)

                record_batch = pa.RecordBatch.from_arrays(
                    [ids, texts, source], names=["id", "text", "source"]
                )
                pqwriter.write_batch(record_batch)
    else:
        print(
            f"Poetry Foundation csv file not found for text compilation (expected at {POETRY_FOUNDATION_CSV})"
        )

    # close the parquet file
    pqwriter.close()


def compile_metadata(data_dir, output_file):
    # for poem dataset output, we need poem id, author, and title
    # to match text results, we need poem id and source id

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("source", pa.string()),
            ("author", pa.string()),
            ("title", pa.string()),
        ]
    )
    # open a parquet writer for outputting content in batches
    pqwriter = pq.ParquetWriter(output_file, schema)

    # load chadwyck healey metadata
    if CHADWYCK_HEALEY_CSV.exists():
        # use polars to read in the csv and convert to the format we want
        # - rename main title to title
        # - add source id for all rows
        # - combine author first and last name
        # - reorder and limit columns to match parquet schema
        df = (
            # ignore parse errors in fields we don't care about (author_dob)
            pl.read_csv(CHADWYCK_HEALEY_CSV, ignore_errors=True)
            .rename({"title_main": "title"})
            .with_columns(source=pl.lit(SOURCE_ID["Chadwyck-Healey"]))
            .with_columns(
                pl.concat_str(
                    [pl.col("author_fname"), pl.col("author_lname")], separator=" "
                ).alias("author")
            )
            .select(["id", "source", "author", "title"])
        )
        # convert polars dataframe to arrow table, cast to our schema to
        # align types (large string vs string), then write out in batches
        for batch in df.to_arrow().cast(target_schema=schema).to_batches():
            pqwriter.write_batch(batch)
    else:
        print(
            f"Chadwyck-Healey csv file not found for metadata compilation (expected at {CHADWYCK_HEALEY_CSV})"
        )

    # load poetry foundation data from csv file
    if POETRY_FOUNDATION_CSV.exists():
        # use polars to read in the csv and convert to the format we want
        # - rename columns to match desired output
        # - add source id
        # - reorder and limit columns to match parquet schema
        df = (
            pl.read_csv(POETRY_FOUNDATION_CSV)
            # .drop("Content", "")
            .rename(
                {"Author": "author", "Title": "title", "Poetry Foundation ID": "id"}
            )
            .with_columns(source=pl.lit(SOURCE_ID["Poetry Foundation"]))
            .select(["id", "source", "author", "title"])
        )
        # convert polars dataframe to arrow table, cast to our schema to
        # align types (large string vs string), then write out in batches
        for batch in df.to_arrow().cast(target_schema=schema).to_batches():
            pqwriter.write_batch(batch)
    else:
        print(
            f"Poetry Foundation csv file not found for metadata compilation (expected at {POETRY_FOUNDATION_CSV})"
        )

    # for the directory of internet poems, metadata is embedded in file name
    internet_poems_dir = data_dir / "internet-poems"
    # this directory is a set of manually curated texts;
    # currently only 112 files, so don't worry about chunking until needed
    poem_files = list(internet_poems_dir.glob("*.txt"))
    # use filename without .txt as poem identifier
    ids = [p.stem for p in poem_files]
    # filename is : Firstname-Lastname_Poem-Title.txt
    # author name: filename before the _ with dashes replaced with spaces
    authors = [p.stem.split("_")[0].replace("-", " ") for p in poem_files]
    # title: same as author for the text after the _
    titles = [p.stem.split("_")[1].replace("-", " ") for p in poem_files]
    source = ["internet-poems"] * len(ids)

    # create a record batch to write out
    record_batch = pa.RecordBatch.from_arrays(
        [ids, source, authors, titles], names=["id", "source", "author", "title"]
    )
    pqwriter.write_batch(record_batch)

    # close the parquet file
    pqwriter.close()


def main():
    # if the parquet files aren't present, generate them
    # (could add an option to recompile in future)
    if not TEXT_PARQUET_FILE.exists():
        print(f"Compiling reference poem text to {TEXT_PARQUET_FILE}")
        compile_text(REF_DATA_DIR, TEXT_PARQUET_FILE)
    if not META_PARQUET_FILE.exists():
        print(f"Compiling reference poem metadata to {META_PARQUET_FILE}")
        compile_metadata(REF_DATA_DIR, META_PARQUET_FILE)

    # test searching
    df = pl.read_parquet(TEXT_PARQUET_FILE)
    meta_df = pl.read_parquet(META_PARQUET_FILE)

    test_strings = [
        "To each sagacious nose apply",
        "complex, ovoid emptiness",
        "o gently guide my pilgrim feet",
        "pursued their flight",
        "roll in safety",
        "as children birds",
        "more shame",
        "fatal arrow; ill coast",
        "what the sex of women",
        "make gods adore",
        "attend to what a lesser muse",
        "that pleasure is the chiefest good",
    ]

    for text in test_strings:
        print(f"\nLooking for '{text}'")
        # case insensitive, multiline
        # todo: ignore/normalize whitespace
        # ... use polars to generate a simplified version to match against
        result = df.filter(pl.col("text").str.contains(f"(?im){text}"))
        print(len(result))
        # if we have one and only one match
        if not result.is_empty():
            # join resulting poems with metadata dataframe
            # so we can include poem title and author in the output
            result = result.join(
                meta_df,
                # join on the combination of poem id and source id
                on=pl.concat_str([pl.col("id"), pl.col("source")], separator="|"),
            )

            for poem in result.to_dicts()[:3]:
                print(
                    f"id:{poem['id']} source:{poem['source']} title:{poem['title']} author:{poem['author']}"
                )
                poem_beginning = poem["text"][:90].replace("\n", " ").strip()
                poem_beginning = re.sub(r"\s+", " ", poem_beginning)
                print(f"   {poem_beginning} ...")


if __name__ == "__main__":
    main()
