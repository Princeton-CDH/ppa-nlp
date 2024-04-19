import sys
import os.path
import gzip
import bz2
import json
import csv
import unicodedata

from collections import Counter
from tqdm import tqdm


def open_jsonl(filename, mode="rt"):
    """
    Opens a possibly compressed .jsonl file
    Returns: file object
    """
    file_ext = os.path.splitext(filename)[1]

    if file_ext == ".jsonl":
        return open(filename, mode=mode)
    elif file_ext == ".gz":
        return gzip.open(filename, mode=mode)
    elif file_ext == ".bz2":
        return bz2.open(filename, mode=mode)
    else:
        print(f"ERROR: Unsupported extension '{file_ext}'")
        sys.exit(1)


def get_char_name(char):
    """
    Returns the name of the unicode character
    """
    if char == "\n":
        return "LINE FEED"
    else:
        # Return N/A if no name is found within the Unicode Character Database
        return unicodedata.name(char, "N/A")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: [input jsonl] [output tsv]")
        sys.exit(1)

    in_jsonl = sys.argv[1]
    out_tsv = sys.argv[2]

    if not os.path.isfile(in_jsonl):
        print(f"ERROR: '{in_jsonl}' does not exist")
        sys.exit(1)
    if os.path.isfile(out_tsv):
        print(f"ERRORL '{out_tsv}' exists")
        sys.exit(1)

    chars = Counter()
    # Get line count for tqdm
    n_lines = sum([1 for line in open_jsonl(in_jsonl, mode="rb")])
    with open_jsonl(in_jsonl) as reader:
        for line in tqdm(reader, total=n_lines):
            page = json.loads(line)
            if "text" in page:
                chars.update(page["text"])

    field_names = ["char", "unicode codepoint", "unicode name", "count"]
    with open(out_tsv, mode="w", newline="") as file_handler:
        writer = csv.DictWriter(
            file_handler, dialect="excel-tab", fieldnames=field_names
        )
        writer.writeheader()
        for char in chars:
            char_repr = repr(char)
            entry = {
                "char": char if char_repr[1] == char else char_repr,
                "unicode codepoint": f"{ord(char):x}",
                "unicode name": get_char_name(char),
                "count": chars[char],
            }
            writer.writerow(entry)
