"""
Script for collectin character-level statistics

env: ppa-ocr
"""

import sys
import os.path
import json
import csv
import unicodedata

from collections import Counter
from tqdm import tqdm
from helper import open_jsonl
from ocr_helper import clean_chars


__cc_names = {
    "\n": "Cc: LINE FEED",
    "\t": "Cc: TAB",
    "\r": "Cc: CARRIAGE RETURN",
    "\u007f": "Cc: DEL",
    "\u0080": "Cc: PAD",
    "\u0084": "Cc: IND",
    "\u0085": "Cc: NEL",
    "\u0086": "Cc: SSA",
    "\u008e": "Cc: SS2",
    "\u008f": "Cc: SS3",
    "\u0094": "Cc: CCH",
    "\u0095": "Cc: MW",
    "\u0099": "Cc: SGC",
    "\u009c": "Cc: ST",
}


def get_char_name(char):
    """
    Returns the name of the unicode character
    """
    # Check if control character
    if unicodedata.category(char) == "Cc":
        return __cc_names.get(char, "Cc: {ord(char):04x}")
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

    count_data = {
        "raw": {"f": Counter(), "df": Counter(), "wf": Counter()},
        "clean": {"f": Counter(), "df": Counter(), "wf": Counter()},
    }
    work_chars = {"raw": {}, "clean": {}}

    # Get line count for tqdm
    n_lines = sum([1 for line in open_jsonl(in_jsonl, mode="rb")])
    with open_jsonl(in_jsonl) as reader:
        for line in tqdm(reader, total=n_lines):
            page = json.loads(line)
            if "text" in page:
                work = page["work_id"]
                # Get raw & clean page-level counts
                counts = {
                    "raw": Counter(page["text"]),
                    "clean": Counter(clean_chars(page["text"])),
                }
                for tmt in ["raw", "clean"]:
                    # Update count data
                    count_data[tmt]["f"] += counts[tmt]
                    count_data[tmt]["df"].update(counts[tmt].keys())
                    # Update work-level character sets
                    if work in work_chars[tmt]:
                        work_chars[tmt][work] |= counts[tmt].keys()
                    else:
                        work_chars[tmt][work] = set(counts[tmt].keys())

    # Build work-level freqs (wf)
    for tmt in ["raw", "clean"]:
        for work, char_set in work_chars[tmt].items():
            count_data[tmt]["wf"].update(char_set)

    field_names = [
        "char",
        "unicode codepoint (hex)",
        "unicode codepoint (dec)",
        "unicode name",
        "raw f",
        "raw df",
        "raw wf",
        "clean f",
        "clean df",
        "clean wf",
    ]
    with open(out_tsv, mode="w", newline="") as file_handler:
        writer = csv.DictWriter(
            file_handler, dialect="excel-tab", fieldnames=field_names
        )
        writer.writeheader()
        for char in count_data["raw"]["f"]:
            char_repr = repr(char)
            entry = {
                "char": char if char_repr[1] == char else char_repr,
                "unicode codepoint (hex)": f"{ord(char):04x}",
                "unicode codepoint (dec)": f"{ord(char)}",
                "unicode name": get_char_name(char),
            }
            for tmt in ["raw", "clean"]:
                entry[f"{tmt} f"] = count_data[tmt]["f"][char]
                entry[f"{tmt} df"] = count_data[tmt]["df"][char]
                entry[f"{tmt} wf"] = count_data[tmt]["wf"][char]

            writer.writerow(entry)
