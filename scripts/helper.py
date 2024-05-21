"""
Library of general-purpose auxiliary methods for stand-alone scripts
"""

import os
import sys
import gzip
import bz2


_char_conversion_map = {"ſ": "s"}
_char_translation_table = str.maketrans(_char_conversion_map)


def clean_htid(htid):
    """
    Returns the "clean" version of a HathiTrust volume id
    """
    lib_id, vol_id = htid.split(".", 1)
    vol_id = vol_id.replace(":", "+")
    vol_id = vol_id.replace("/", "=")
    vol_id = vol_id.replace(".", ",")
    return f"{lib_id}.{vol_id}"


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
