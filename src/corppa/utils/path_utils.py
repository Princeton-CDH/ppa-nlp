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


def get_ppa_source(vol_id):
    """
    For a given volume id, return the corresponding source.
    Assume:
        * Gale volume ids begin with "CW0" or "CBO"
        * Hathitrust volume ids contain a "."
    """
    # Note that this is fairly brittle.
    if vol_id.startswith("CW0") or vol_id.startswith("CB0"):
        return "Gale"
    elif "." in vol_id:
        return "HathiTrust"
    else:
        raise ValueError(f"Can't identify source for volume '{vol_id}'")


def get_stub_dir(source, vol_id):
    """
    Returns the stub directory name for the specified volume (vol_id) and
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


def get_vol_dir(vol_id):
    """
    Returns the volume directory (pathlib.Path) for the specified volume (vol_id)
    """
    source = get_ppa_source(vol_id)
    if source == "Gale":
        return pathlib.Path(source, get_stub_dir(source, vol_id), vol_id)
    elif source == "HathiTrust":
        # TODO: This does not match tigerdata
        # return pathlib.Path(source, get_stub_dir(source, vol_id), encode_htid(vol_id))
        raise NotImplementedError(f"{source} volume directory conventions TBD")
    else:
        raise ValueError(f"Unknown source '{source}'")


def get_volume_id(work_id):
    """
    Extract volume id from PPA work id

    * For full works, volume ids and work ids are the same.
    * For excerpts, the work id is composed of the prefix followed by "-p" and
      the starting page of the excerpt.
    """
    return work_id.rsplit("-p", 1)[0]


def get_image_relpath(work_id, page_num):
    """
    Get the (relative) image path for specified PPA work page
    """
    vol_id = get_volume_id(work_id)
    vol_dir = get_vol_dir(vol_id)
    source = get_ppa_source(vol_id)
    if source == "Gale":
        image_name = f"{vol_id}_{page_num:04d}0.TIF"
        return vol_dir.joinpath(image_name)
    elif source == "HathiTrust":
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported source '{source}'")
