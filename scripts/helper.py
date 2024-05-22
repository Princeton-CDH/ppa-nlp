"""
Library of general-purpose auxiliary methods for stand-alone scripts
"""

import os
import sys
import gzip
import bz2


_htid_encode_map = {":": "+", "/": "=", ".": ","}
_htid_encode_table = str.maketrans(_htid_encode_map)
_htid_decode_map = {v: k for k, v in _htid_encode_map.items()}
_htid_decode_table = str.maketrans(_htid_decode_map)


def encode_htid(htid):
    """
    Returns the "clean" version of a HathiTrust volume identifier with the form:
        [library id].[volume id]
    Specifically, the volume-portion of the id undergoes the following
    character replacement: ":" --> "+", "/" --> "=", "." --> ","
    """
    lib_id, vol_id = htid.split(".", 1)
    vol_id = vol_id.translate(_htid_encode_table)
    return f"{lib_id}.{vol_id}"


def decode_htid(clean_htid):
    """
    Return original HathiTrust volume identifier from clean (encoded) version:
        [library id].[encoded volume id]
    Specifically, the volume-portion of the id undergoes the following
    character replacement: "+" --> ":", "=" --> "/", "," --> "."
    """
    lib_id, vol_id = clean_htid.split(".", 1)
    vol_id = vol_id.translate(_htid_decode_table)
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
