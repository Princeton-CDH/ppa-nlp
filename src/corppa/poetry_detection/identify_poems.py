#!/usr/bin/env python
"""

Script to identify poem excerpts by matching against a local
collection of reference poems.

Setup:

Download and extract poetry-ref-data.tar.bz2 from /tigerdata/cdh/prosody/poetry-detection
You should extract it in the same directory where you plan to run this script.
The script will compile reference content into full-text and metadata parquet files;
to recompile, rename or remove the parquet files.

"""

import argparse
import csv
import logging
import pathlib
import re
from glob import iglob
from itertools import batched
from time import perf_counter

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import rapidfuzz
from tqdm import tqdm
from unidecode import unidecode

logger = logging.getLogger(__name__)

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
        # load poetry foundation csv into a polars dataframe
        # - rename columns for our use
        # - add source column
        # - select only the columns we want to include
        pf_df = (
            pl.read_csv(POETRY_FOUNDATION_CSV)
            .rename({"Poetry Foundation ID": "id", "Content": "text"})
            .with_columns(source=pl.lit(SOURCE_ID["Poetry Foundation"]))
            .select(["id", "text", "source"])
        )
        # convert polars dataframe to arrow table, cast to our schema to
        # align types (large string vs string), then write out in batches
        for batch in pf_df.to_arrow().cast(target_schema=schema).to_batches():
            pqwriter.write_batch(batch)
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

    # for the directory of internet poems, metadata is embedded in file name
    internet_poems_dir = data_dir / "internet-poems"
    # this directory is a set of manually curated texts;
    # currently only 112 files, so don't worry about chunking until needed
    poem_files = list(internet_poems_dir.glob("*.txt"))
    # use filename without .txt as poem identifier
    ids = [p.stem for p in poem_files]
    # filename is : Firstname-Lastname_Poem-Title.txt
    # author name: filename before the _ with dashes replaced with spaces
    authors = [p.stem.split("_", 1)[0].replace("-", " ") for p in poem_files]
    # title: same as author for the text after the _
    titles = [p.stem.split("_", 1)[1].replace("-", " ") for p in poem_files]
    source = ["internet-poems"] * len(ids)

    # create a record batch to write out
    record_batch = pa.RecordBatch.from_arrays(
        [ids, source, authors, titles], names=["id", "source", "author", "title"]
    )
    pqwriter.write_batch(record_batch)

    # load poetry foundation data from csv file
    # do this one last since it is least preferred of our sources
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

    # close the parquet file
    pqwriter.close()


def _text_for_search(expr):
    """Takes a polars expression (e.g. column or literal value) and applies
    text pattern replacements to clean up to make it easier to find matches."""
    return (
        # remove specific punctuation marks in the middle of words
        expr.str.replace_all(r"(\w)[-'](\w)", "$1$2")
        # metrical notation that splits words (e.g. sudden | -ly or visit | -or)
        .str.replace_all(r"(\w) \| -(\w)", "$1$2")
        # replace other puncutation with spaces
        .str.replace_all("[[:punct:]]", " ")
        # normalize whitespace
        .str.replace_all("[\t\n\v\f\r ]+", " ")  # could also use [[:space:]]
        # replace curly quotes with straight (both single and double)
        .str.replace_all("[”“]", '"')
        .str.replace_all("[‘’]", "'")
        .str.replace_all("ſ", "s")  # handle long s (also handled by unidecode)
        .str.strip_chars()
    )


def searchable_text(text):
    """Convert a text string into a searchable string, using the same rules
    applied to search text in the reference dataframe, with additional unicode decoding.
    """
    return unidecode(pl.select(_text_for_search(pl.lit(text))).item())


def generate_search_text(df, field="text"):
    """Takes a Polars dataframe and generates a searchable text field
    based on the input column (by default, input column is "text" and
    output is "search_text").  Removes punctuation, normalizes whitespace,
    and strips trailing and leading whitespace, etc."""
    output_field = f"search_{field}"
    return df.with_columns(**{output_field: _text_for_search(pl.col(field))})


def multiple_matches(df, search_field):
    # when a result has multiple matches, see if we can determine if
    # it is the same poem in different sources
    match_count = int(df.height)

    #  check if both author and title match (ignoring punctuation and case)
    # TODO: use rapidfuzz here to check author & title are sufficiently similar
    # e.g. these should be treated as matches but are not currently:
    #    Walter Scott      ┆ Coronach
    #    Walter, Sir Scott ┆ CCLXXVIII CORONACH
    df = df.with_columns(
        _author=pl.col("author").str.replace_all("[[:punct:]]", "").str.to_lowercase(),
        _title=pl.col("title").str.replace_all("[[:punct:]]", "").str.to_lowercase(),
    )

    dupe_df = df.filter(df.select(["_author", "_title"]).is_duplicated())

    match_poem = None
    if not dupe_df.is_empty():
        # if all rows match, return the first one
        if int(dupe_df.height) == match_count:
            match_poem = dupe_df.to_dicts()[0]
            match_poem["notes"] = (
                f"multiple matches on {search_field}, all rows match author + title"
            )
        # if duplicate rows are a majority, return the first one
        elif dupe_df.height >= match_count / 2:
            # TODO: include alternates in notes?
            # these majority matches may be less confident
            match_poem = dupe_df.to_dicts()[0]
            match_poem["notes"] = (
                f"multiple matches on {search_field}, majority match author + title ({dupe_df.height} out of {match_count})"
            )

    if match_poem is not None:
        return match_poem

    # if author/title duplication check failed, check for author matches
    # poetry foundation includes shakespeare drama excerpts with alternate names
    authordupe_df = df.filter(df.select(["_author"]).is_duplicated())
    if not authordupe_df.is_empty():
        # shakespeare shows up oddly in poetry foundation;
        # if author matchnes assume the other source has the correct title
        non_poetryfoundtn = authordupe_df.filter(
            pl.col("source") != SOURCE_ID["Poetry Foundation"]
        )
        if non_poetryfoundtn.height == 1:
            match_poem = non_poetryfoundtn.to_dicts()[0]
            match_poem["notes"] = (
                "multiple matches, duplicate author but not title; excluding Poetry Foundation"
            )
            return match_poem


def fuzzy_partial_ratio(series, search_text):
    """Calculate rapidfuzz partial_ratio score across for a single input
    search text across a whole series of potentially matching texts.
    Returns a list of scores."""
    scores = rapidfuzz.process.cdist(
        [search_text],
        series,
        scorer=rapidfuzz.fuzz.partial_ratio,
        score_cutoff=90,
        workers=-1,
    )
    # generates a list of scores for each input search text, but we only have
    # one input string, so return the first list of scores
    return scores[0]


def find_reference_poem(ref_df, input_row, meta_df):
    result = {"poem_id"}
    for search_field in ["search_text", "search_first_line", "search_last_line"]:
        # use unidecode to drop accents (often used to indicate meter)
        search_text = unidecode(input_row[search_field])

        try:
            # do a case-insensitive search
            result = ref_df.filter(
                pl.col("search_text").str.contains(f"(?i){search_text}")
            )
        except pl.exceptions.ComputeError as err:
            print(f"Error searching: {err}")
            continue
        # if no matches, try the next search field
        if result.is_empty():
            continue

        # otherwise, see if the results are useful
        num_matches = result.height  # height = number of rows
        # join matching poem with metadata dataframe
        # so we can include poem title and author in the output
        result = result.join(
            meta_df,
            # join on the combination of poem id and source id
            on=pl.concat_str([pl.col("id"), pl.col("source")], separator="|"),
            how="left",  # occasionally ids do not match,
            # e.g. Chadwyck Healey poem id we have text for but not in metadata
        )

        search_field_label = search_field.replace("search_", "").replace("_", " ")

        # if we get a single match, assume it is authoritative
        if num_matches == 1:
            # match poem includes id, author, title
            match_poem = result.to_dicts()[0]
            # add note about how the match was determined
            match_poem["notes"] = f"single match on {search_field_label}"
            # include number of matches found
            match_poem["num_matches"] = num_matches

            return match_poem
        elif num_matches < 10:
            # if there's a small number of matches, check for duplicates
            match_poem = multiple_matches(result, search_field_label)
            # return match if a good enough result was found
            if match_poem:
                match_poem["num_matches"] = num_matches
                return match_poem

    # if no matches were found yet, try a fuzzy search on the full text
    search_text = unidecode(input_row["search_text"])
    # TODO: may want to require some minimum length or uniqueness on the search text
    logger.debug(f"🔎 Trying fuzzy match on {search_text}")
    start_time = perf_counter()
    result = ref_df.with_columns(
        score=pl.col("search_text").map_batches(
            lambda x: fuzzy_partial_ratio(x, search_text)
        )
    ).filter(pl.col("score").ne(0))
    end_time = perf_counter()
    logger.debug(f"Calculated rapidfuzz partial_ratio in {end_time - start_time:0.2f}s")

    result = result.join(
        meta_df,
        # join on the combination of poem id and source id
        on=pl.concat_str([pl.col("id"), pl.col("source")], separator="|"),
        how="left",
    )
    result = result.drop("text", "id_right", "source_right", "search_text")
    num_matches = result.height

    result = result.sort(by="score", descending=True)
    if not result.is_empty():
        # when we only get a single match, results look pretty good
        if num_matches == 1:
            # match poem includes id, author, title
            match_poem = result.to_dicts()[0]
            # add note about how the match was determined
            match_poem["notes"] = f"fuzzy match; score: {result['score'].item():.1f}"
            # include number of matches found
            match_poem["num_matches"] = num_matches
            return match_poem
        elif num_matches <= 3:
            # if there's a small number of matches, check for duplicates
            match_poem = multiple_matches(result, "full text (fuzzy)")
            # return match if a good enough result was found
            if match_poem:
                match_poem["num_matches"] = num_matches
                match_poem["notes"] += (
                    f"\nfuzzy match; score: {match_poem['score']:.1f}"
                )
                return match_poem
        else:
            # sometimes we get many results with 100 scores;
            # likely an indication that the search text is short and too common
            # filter to all results with the max score and check for a majority
            top_matches = result.filter(pl.col("score").eq(pl.col("score").max()))
            match_poem = multiple_matches(top_matches, "full text (fuzzy)")
            # return match if a good enough result was found
            if match_poem:
                match_poem["num_matches"] = num_matches
                match_poem["notes"] += f"\nfuzzy match, score {match_poem['score']}"
                return match_poem

    # no good match found
    return None


def main(input_file):
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    # if the parquet files aren't present, generate them
    # (could add an option to recompile in future)
    if not TEXT_PARQUET_FILE.exists():
        print(f"Compiling reference poem text to {TEXT_PARQUET_FILE}")
        compile_text(REF_DATA_DIR, TEXT_PARQUET_FILE)
    if not META_PARQUET_FILE.exists():
        print(f"Compiling reference poem metadata to {META_PARQUET_FILE}")
        compile_metadata(REF_DATA_DIR, META_PARQUET_FILE)

    # load for searching
    df = pl.read_parquet(TEXT_PARQUET_FILE)
    meta_df = pl.read_parquet(META_PARQUET_FILE)
    print(f"Poetry reference text data: {df.height:,} entries")
    # some texts from poetry foundation and maybe Chadwyck-Healey are truncated
    # discard them to avoid bad partial/fuzzy matches
    df = df.with_columns(text_length=pl.col("text").str.len_chars())
    min_length = 30
    short_texts = df.filter(pl.col("text_length").lt(min_length))
    df = df.filter(pl.col("text_length").ge(min_length))
    print(f"  Omitting {short_texts.height} poems with text length < {min_length}")

    print(f"Poetry reference metadata:  {meta_df.height:,} entries")

    # generate a simplified text field for searching
    df = generate_search_text(df)

    input_df = pl.read_csv(input_file)  # .drop("start", "end", "laure's links")
    print(f"Input file has {input_df.height:,} entries")
    # testing against Mary's manual identification
    input_df = (
        input_df.filter(pl.col("author") != "").drop("start", "end", "laure's links")
        # .limit(n=5000)  # limit for now, for testing/review
    )
    print(f"limiting to {input_df.height} rows with metadata to compare")

    # convert input text to search text using the same rules applied to reference df
    input_df = generate_search_text(input_df)
    # split out text to isolate first and last lines
    input_df = input_df.with_columns(
        first_line=pl.col("text").str.split("\n").list.first(),
        last_line=pl.col("text").str.split("\n").list.last(),
    )
    # generate searchable versions of first and last lines
    input_df = generate_search_text(input_df, "first_line")
    input_df = generate_search_text(input_df, "last_line")

    # create lists for content that will go in output columns
    match_count = []
    poem_id = []
    author = []
    title = []
    notes = []

    # number of matches found
    match_found = 0

    # wrap the row iterator in a tqdm progress bar
    progress_poems = tqdm(
        # use polars iter_rows with names to get a dictionary for each entry
        input_df.iter_rows(named=True),
        desc="Identifying...",
        bar_format="{desc} processed {n:,} poems{postfix} | elapsed: {elapsed}",
        # disable=disable_progress,
    )

    for i, row in enumerate(progress_poems):
        match_poem = find_reference_poem(df, row, meta_df)
        if match_poem:
            poem_id.append(match_poem["id"])
            author.append(match_poem.get("author"))
            title.append(match_poem.get("title"))
            match_count.append(match_poem["num_matches"])
            notes.append(match_poem.get("notes"))

            # update the tally of rows we found matches for
            match_found += 1
            # update report in the progress bar
            progress_poems.set_postfix_str(
                f"matched {match_found:,} ({match_found / i:.2f}%)"
            )

        else:
            # if no match was found, add empty rows to the columns
            match_count.append(0)  # integer column
            for col in poem_id, author, title, notes:
                col.append("")  # string columns\

    # augment filtered input with output and save to file
    output_df = input_df.with_columns(
        match_count=pl.Series(match_count),
        match_poem_id=pl.Series(poem_id),
        match_author=pl.Series(author),
        match_title=pl.Series(title),
        match_notes=pl.Series(notes),
    ).drop(
        "search_text",
        "search_first_line",
        "search_last_line",
        "first_line",
        "last_line",
    )
    output_file = input_file.with_name(f"{input_file.stem}_matched.csv")
    # TODO: figure out how to write byte-order-mark to indicate unicode
    output_df.write_csv(output_file)
    print(f"Poems with match information saved to {output_file}")
    print(
        f"{match_found} excerpts with matches ({match_found / input_df.height * 100:.2f}% of {input_df.height} rows processed)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attempt to identify poem excerpts by matching against reference set"
    )
    parser.add_argument(
        "input",
        # TODO: determine minimum required fields; maybe just text?
        help="csv or tsv file with poem excerpts",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    main(args.input)
