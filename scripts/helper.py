"""
Library of auxiliary methods for scripts

env: ppa-ocr
"""

import os
import sys
import gzip
import bz2
import ftfy


_char_conversion_map = {"ſ": "s"}
_char_translation_table = str.maketrans(_char_conversion_map)


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


def clean_chars(text):
    """
    Initial cleaning of text focused on characters.
    """
    result = ftfy.fix_text(
        text,
        unescape_html=False,
        fix_encoding=False,
        normalization="NFC",
        explain=False,
    )
    result = result.translate(_char_translation_table)
    return result
