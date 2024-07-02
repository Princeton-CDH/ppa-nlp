"""
Library of general-purpose auxiliary methods for stand-alone scripts
"""

_htid_encode_map = {":": "+", "/": "=", ".": ","}
_htid_encode_table = str.maketrans(_htid_encode_map)
_htid_decode_map = {v: k for k, v in _htid_encode_map.items()}
_htid_decode_table = str.maketrans(_htid_decode_map)


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
