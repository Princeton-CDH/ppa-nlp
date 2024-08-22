"""
Library of general-purpose auxiliary methods for stand-alone scripts
"""

import pathlib

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
    if "." not in htid:
        raise ValueError(f"Invalid htid '{htid}'")
    lib_id, vol_id = htid.split(".", 1)
    vol_id = vol_id.translate(_htid_encode_table)
    return f"{lib_id}.{vol_id}"


def decode_htid(encoded_htid):
    """
    Return original HathiTrust volume identifier from encoded version:
        [library id].[encoded volume id]
    Specifically, the volume-portion of the id undergoes the following
    character replacement: "+" --> ":", "=" --> "/", "," --> "."
    """
    if "." not in encoded_htid:
        raise ValueError(f"Invalid encoded htid '{encoded_htid}'")
    lib_id, vol_id = encoded_htid.split(".", 1)
    vol_id = vol_id.translate(_htid_decode_table)
    return f"{lib_id}.{vol_id}"


def get_stub_dir(source, vol_id):
    """
    Returns the stub directory for the specified volume (vol_id) and
    source type (source)

    For Gale, every third number (excluding the leading 0) of the volume
    identifier is used.
       Ex. CB0127060085 --> 100

    For HathiTrust, the library portion of the volume identifier is used.
        Ex. mdp.39015003633594 --> mdp
    """
    if source == "Gale":
        return vol_id[::3][1:]
    elif source == "HathiTrust":
        return vol_id.split(".", maxsplit=1)[0]
    else:
        raise ValueError(f"Unknown source '{source}'")


def get_vol_dir(source, vol_id):
    if source == "Gale":
        return pathlib.Path(source, get_stub_dir(source, vol_id), vol_id)
    elif source == "HathiTrust":
        return pathlib.Path(source, get_stub_dir(source, vol_id), encode_htid(vol_id))
    else:
        raise ValueError(f"Unknown source '{source}'")
