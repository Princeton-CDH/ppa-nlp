"""
General-purpose methods for working with paths, PPA identifiers, and directories
"""

import os
import pathlib
from typing import Iterator

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


def find_relative_paths(
    base_dir, exts, follow_symlinks=True, group_by_dir=False
) -> Iterator[pathlib.Path] | Iterator[tuple[pathlib.Path, list]]:
    """
    This method finds files anywhere under the specified base directory
    that match any of the specified file extensions (case insensitive),
    and returns a generator of path objects with a path relative to the
    base directory. File extensions should include the leading period,
    i.e. `[".jpg", ".tiff"]` rather than `["jpg", "tiff"]`.

    For example, given a base directory `a/b/c/images`, an extension list of `.jpg`,
    and files nested at different levels in the hierarchy
    `a/b/c/images/alpha.jpg`, `a/b/c/images/d/beta.jpg`:
    ```
    a/b/c/images
      |-- alpha.jpg
      +-- d
          |-- beta.jpg
    ```
    The result will include the two items: `alpha.jpg and `d/beta.jpg`

    When `group_by_dir` is `True`, resulting files will be returned grouped
    by the parent directory. The return result is a tuple of a single :class:`pathlib.Path`
    object for the directory and a list of :class:`pathlib.Path` objects for the files in that
    directory that match the specified extensions.  Given a hierarchy like this:
    ```
    images/vol-a/
      |-- alpha.jpg
      |-- beta.jpg
    ```
    the method would return `(vol-a, [alpha.jpg, beta.jpg])`.
    """
    # Create lowercase extension set from passed in exts
    ext_set = {ext.lower() for ext in exts}

    # Using pathlib.Path.walk / os.walk over glob because (1) it allows us to
    # find files with multiple extensions in a single walk of the directory
    # and (2) lets us leverage additional functionality of pathlib.
    if hasattr(base_dir, "walk"):
        # As of Python 3.12, Path.walk exists
        walk_generator = base_dir.walk(follow_symlinks=follow_symlinks)
    else:
        # For Python 3.11, fall back to os.walk
        walk_generator = os.walk(base_dir, followlinks=follow_symlinks)
    for dirpath, dirnames, filenames in walk_generator:
        if isinstance(dirpath, str):
            # Convert str produced by os.walk to Path object
            dirpath = pathlib.Path(dirpath)
        # Create a generator of relevant files in the current directory
        include_files = (
            dirpath.joinpath(file).relative_to(base_dir)
            for file in filenames
            if os.path.splitext(file)[1].lower() in ext_set
        )
        # if group by dir is specified, yield dirpath and list of files,
        # but only if at least one relevant file is found
        if group_by_dir:
            include_files = list(include_files)
            if include_files:
                yield (dirpath.relative_to(base_dir), include_files)
        else:
            # otherwise yield just the files
            yield from include_files

        # modify dirnames in place to skip hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
