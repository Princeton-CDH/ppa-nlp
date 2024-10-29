import pathlib
from types import GeneratorType
from unittest.mock import patch

import pytest

from corppa.utils.path_utils import (
    decode_htid,
    encode_htid,
    find_relative_paths,
    get_image_relpath,
    get_ppa_source,
    get_stub_dir,
    get_vol_dir,
    get_volume_id,
)


def test_encode_htid():
    assert encode_htid("mdp.39015003633594") == "mdp.39015003633594"
    assert encode_htid("dul1.ark:/13960/t5w67998k") == "dul1.ark+=13960=t5w67998k"
    assert encode_htid("miun.aaa3406.0001.001") == "miun.aaa3406,0001,001"
    with pytest.raises(ValueError, match="Invalid htid 'xxx0000'"):
        encode_htid("xxx0000")


def test_decode_htid():
    assert decode_htid("mdp.39015003633594") == "mdp.39015003633594"
    assert decode_htid("dul1.ark+=13960=t5w67998k") == "dul1.ark:/13960/t5w67998k"
    assert decode_htid("miun.aaa3406,0001,001") == "miun.aaa3406.0001.001"
    with pytest.raises(ValueError, match="Invalid encoded htid 'xxx0000'"):
        decode_htid("xxx0000")


def test_encode_decode_htid():
    assert decode_htid(encode_htid("mdp.39015003633594")) == "mdp.39015003633594"
    assert (
        decode_htid(encode_htid("dul1.ark:/13960/t5w67998k"))
        == "dul1.ark:/13960/t5w67998k"
    )

    assert decode_htid(encode_htid("miun.aaa3406.0001.001")) == "miun.aaa3406.0001.001"


def test_get_ppa_source():
    assert get_ppa_source("CB0127060085") == "Gale"
    assert get_ppa_source("CW0116527364") == "Gale"
    assert get_ppa_source("mdp.39015010540071") == "HathiTrust"
    with pytest.raises(ValueError, match="Can't identify source for volume 'xxx0000'"):
        get_ppa_source("xxx0000")


def test_get_stub_dir():
    # Gale
    assert get_stub_dir("Gale", "CB0127060085") == "100"
    # HathiTrust
    assert get_stub_dir("HathiTrust", "mdp.39015003633594") == "mdp"
    # Other
    with pytest.raises(ValueError, match="Unknown source 'invalid src'"):
        get_stub_dir("invalid src", "xxx0000")


@patch("corppa.utils.path_utils.get_stub_dir", return_value="stub_name")
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_vol_dir_gale(mock_get_ppa_source, mock_get_stub_dir):
    # Set returned source value to Gale
    mock_get_ppa_source.return_value = "Gale"
    assert get_vol_dir("gale_id") == pathlib.Path("Gale", "stub_name", "gale_id")
    mock_get_ppa_source.assert_called_with("gale_id")
    mock_get_stub_dir.assert_called_with("Gale", "gale_id")


@patch("corppa.utils.path_utils.get_stub_dir", return_value="stub_name")
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_vol_dir_hathi(mock_get_ppa_source, mock_get_stub_dir):
    # Set returned source value to HathiTrust
    mock_get_ppa_source.return_value = "HathiTrust"
    # TODO: Update once HathiTrust directory conventions are finalized
    with pytest.raises(
        NotImplementedError, match="HathiTrust volume directory conventions TBD"
    ):
        get_vol_dir("htid")
    mock_get_ppa_source.assert_called_with("htid")
    mock_get_stub_dir.assert_not_called()


@patch("corppa.utils.path_utils.get_stub_dir", return_value="stub_name")
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_vol_dir_unk(mock_get_ppa_source, mock_get_stub_dir):
    # Set returned source value
    mock_get_ppa_source.return_value = "Unknown"
    with pytest.raises(ValueError, match="Unknown source 'Unknown'"):
        get_vol_dir("vol_id")
    mock_get_ppa_source.assert_called_with("vol_id")
    mock_get_stub_dir.assert_not_called()


def test_get_volume_id():
    # Full works
    for work_id in ["CB0131351206", "dul1.ark:/13960/t5w67998k"]:
        assert get_volume_id(work_id) == work_id

    # Excerpts
    assert get_volume_id("CW0102294490-pxvi") == "CW0102294490"
    assert get_volume_id("coo1.ark:/13960/t4bp0n867-p3") == "coo1.ark:/13960/t4bp0n867"


@patch("corppa.utils.path_utils.get_volume_id", return_value="vol_id")
@patch("corppa.utils.path_utils.get_vol_dir", return_value=pathlib.Path("vol_dir"))
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_image_relpath(mock_get_ppa_source, mock_get_vol_dir, mock_get_volume_id):
    # Gale
    mock_get_ppa_source.return_value = "Gale"
    assert get_image_relpath("test_id", 4) == pathlib.Path(
        "vol_dir", "vol_id_00040.TIF"
    )
    assert get_image_relpath("test_id", 100) == pathlib.Path(
        "vol_dir", "vol_id_01000.TIF"
    )

    # HathiTrust
    mock_get_ppa_source.return_value = "HathiTrust"
    with pytest.raises(NotImplementedError):
        get_image_relpath("test_id", 4)

    # Other sources
    mock_get_ppa_source.return_value = "EEBO"
    with pytest.raises(ValueError, match="Unsupported source 'EEBO'"):
        get_image_relpath("test_id", 4)


def test_find_relative_paths(tmp_path):
    jpg_a = pathlib.Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    txt_b = pathlib.Path("b.txt")
    tmp_path.joinpath(txt_b).touch()

    # I. Single ext
    paths = find_relative_paths(tmp_path, [".jpg"])
    assert isinstance(paths, GeneratorType)
    assert [jpg_a] == list(paths)

    # II. Multiple extensions
    tif_c = pathlib.Path("c.tif")
    tmp_path.joinpath(tif_c).touch()
    paths = list(find_relative_paths(tmp_path, [".jpg", ".tif"]))
    assert {jpg_a, tif_c} == set(paths)

    # III. Extension handling is case insensitive
    jpg_d = pathlib.Path("d.JPG")
    tmp_path.joinpath(jpg_d).touch()
    paths_a = list(find_relative_paths(tmp_path, [".jpg"]))
    paths_b = list(find_relative_paths(tmp_path, [".JPG"]))
    assert set(paths_a) == set(paths_b)
    assert {jpg_a, jpg_d} == set(paths_a)


def test_find_relative_paths_nested(tmp_path):
    img_dir = tmp_path.joinpath(tmp_path, "images")
    img_dir.mkdir()
    jpg_a = pathlib.Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    jpg_b = pathlib.Path("b.jpg")
    img_dir.joinpath(jpg_b).touch()

    paths = find_relative_paths(img_dir, [".jpg"])
    assert {jpg_b} == set(paths)

    paths = find_relative_paths(tmp_path, [".jpg"])
    assert {jpg_a, pathlib.Path("images", "b.jpg")} == set(paths)


def test_image_relpath_hidden_dirs(tmp_path):
    jpg_a = pathlib.Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    hidden_dir = tmp_path.joinpath(".hidden")
    hidden_dir.mkdir()
    jpg_b = pathlib.Path("b.jpg")
    hidden_dir.joinpath(jpg_b).touch()

    paths = list(find_relative_paths(tmp_path, [".jpg"]))
    assert [jpg_a] == paths


def test_find_relative_paths_symbolic_links(tmp_path):
    """
    Test directory sturcture:
     dir_a:
        a.jpg
        b.jpg (symbolic link, file: dir_b/b.jpg)
        dir_c (symbolic link, dir: dir_b/dir_c)
     dir_b:
        b.jpg
        dir_c:
          c.jpg
    """
    # Create directories
    dir_a = tmp_path.joinpath("dir_a")
    dir_a.mkdir()
    dir_b = tmp_path.joinpath("dir_b")
    dir_b.mkdir()
    dir_c = dir_b.joinpath("dir_c")
    dir_c.mkdir()
    # Create files
    jpg_a = pathlib.Path("a.jpg")
    dir_a.joinpath(jpg_a).touch()
    jpg_b = pathlib.Path("b.jpg")
    dir_b.joinpath(jpg_b).touch()
    jpg_c = pathlib.Path("c.jpg")
    dir_c.joinpath(jpg_c).touch()
    # Create symbolic links
    sym_b = dir_a.joinpath("b.jpg")
    sym_b.symlink_to(jpg_b)
    sym_c = dir_a.joinpath("c")
    sym_c.symlink_to(dir_c, target_is_directory=True)

    # Default follows symbolic links
    paths = list(find_relative_paths(dir_a, [".jpg"]))
    assert {jpg_a, jpg_b, pathlib.Path("c", "c.jpg")} == set(paths)

    # Do not follow symbolic links
    paths = list(find_relative_paths(dir_a, [".jpg"], follow_symlinks=False))
    assert {jpg_a, jpg_b} == set(paths)
