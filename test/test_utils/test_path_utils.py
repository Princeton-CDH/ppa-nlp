from pathlib import Path
from types import GeneratorType
from unittest.mock import patch

import pytest

from corppa.utils.path_utils import (
    decode_htid,
    encode_htid,
    find_relative_paths,
    get_image_relpath,
    get_page_number,
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
    assert get_stub_dir("Gale", "CB0127060085") == Path("100")
    # HathiTrust
    assert get_stub_dir("HathiTrust", "mdp.39015003633594") == Path("mdp", "31039")
    # Other
    with pytest.raises(ValueError, match="Unknown source 'invalid src'"):
        get_stub_dir("invalid src", "xxx0000")


@patch("corppa.utils.path_utils.get_stub_dir", return_value="stub_name")
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_vol_dir_gale(mock_get_ppa_source, mock_get_stub_dir):
    # Set returned source value to Gale
    mock_get_ppa_source.return_value = "Gale"
    assert get_vol_dir("gale_id") == Path("Gale", "stub_name", "gale_id")
    mock_get_ppa_source.assert_called_once_with("gale_id")
    mock_get_stub_dir.assert_called_once_with("Gale", "gale_id")


@patch("corppa.utils.path_utils.encode_htid", return_value="encoded_htid")
@patch("corppa.utils.path_utils.get_stub_dir", return_value="stub_name")
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_vol_dir_hathi(mock_get_ppa_source, mock_get_stub_dir, mock_encode_htid):
    # Set returned source value to HathiTrust
    mock_get_ppa_source.return_value = "HathiTrust"
    assert get_vol_dir("htid") == Path("HathiTrust", "stub_name", "encoded_htid")
    mock_get_ppa_source.assert_called_once_with("htid")
    mock_get_stub_dir.assert_called_once_with("HathiTrust", "htid")
    mock_encode_htid.assert_called_once_with("htid")


@patch("corppa.utils.path_utils.get_stub_dir", return_value="stub_name")
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_vol_dir_unk(mock_get_ppa_source, mock_get_stub_dir):
    # Set returned source value
    mock_get_ppa_source.return_value = "Unknown"
    with pytest.raises(ValueError, match="Unknown source 'Unknown'"):
        get_vol_dir("vol_id")
    mock_get_ppa_source.assert_called_once_with("vol_id")
    mock_get_stub_dir.assert_not_called()


def test_get_volume_id():
    # Full works
    for work_id in ["CB0131351206", "dul1.ark:/13960/t5w67998k"]:
        assert get_volume_id(work_id) == work_id

    # Excerpts
    assert get_volume_id("CW0102294490-pxvi") == "CW0102294490"
    assert get_volume_id("coo1.ark:/13960/t4bp0n867-p3") == "coo1.ark:/13960/t4bp0n867"


def test_page_number():
    assert get_page_number(Path("CW0112029406_00180.txt")) == "0018"
    # raise not implemented error if source id is not Gale/ECCO
    with pytest.raises(NotImplementedError):
        assert get_page_number(Path("uva.x002075945_00180.txt")) == "0018"


@patch("corppa.utils.path_utils.get_volume_id", return_value="vol_id")
@patch("corppa.utils.path_utils.get_vol_dir", return_value=Path("vol_dir"))
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_image_relpath(mock_get_ppa_source, mock_get_vol_dir, mock_get_volume_id):
    # Gale
    mock_get_ppa_source.return_value = "Gale"
    assert get_image_relpath("test_id", 4) == Path("vol_dir", "vol_id_00040.TIF")
    assert get_image_relpath("test_id", 100) == Path("vol_dir", "vol_id_01000.TIF")

    # HathiTrust
    mock_get_ppa_source.return_value = "HathiTrust"
    with pytest.raises(NotImplementedError):
        get_image_relpath("test_id", 4)

    # Other sources
    mock_get_ppa_source.return_value = "EEBO"
    with pytest.raises(ValueError, match="Unsupported source 'EEBO'"):
        get_image_relpath("test_id", 4)


def test_find_relative_paths(tmp_path):
    jpg_a = Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    txt_b = Path("b.txt")
    tmp_path.joinpath(txt_b).touch()

    # I. Single ext
    paths = find_relative_paths(tmp_path, [".jpg"])
    assert isinstance(paths, GeneratorType)
    assert [jpg_a] == list(paths)

    # II. Multiple extensions
    tif_c = Path("c.tif")
    tmp_path.joinpath(tif_c).touch()
    paths = list(find_relative_paths(tmp_path, [".jpg", ".tif"]))
    assert {jpg_a, tif_c} == set(paths)

    # III. Extension handling is case insensitive
    jpg_d = Path("d.JPG")
    tmp_path.joinpath(jpg_d).touch()
    paths_a = list(find_relative_paths(tmp_path, [".jpg"]))
    paths_b = list(find_relative_paths(tmp_path, [".JPG"]))
    assert set(paths_a) == set(paths_b)
    assert {jpg_a, jpg_d} == set(paths_a)


def test_find_relative_paths_nested(tmp_path):
    img_dir = tmp_path.joinpath(tmp_path, "images")
    img_dir.mkdir()
    jpg_a = Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    jpg_b = Path("b.jpg")
    img_dir.joinpath(jpg_b).touch()

    paths = find_relative_paths(img_dir, [".jpg"])
    assert {jpg_b} == set(paths)

    paths = find_relative_paths(tmp_path, [".jpg"])
    assert {jpg_a, Path("images", "b.jpg")} == set(paths)


def test_image_relpath_hidden_dirs(tmp_path):
    jpg_a = Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    hidden_dir = tmp_path.joinpath(".hidden")
    hidden_dir.mkdir()
    jpg_b = Path("b.jpg")
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
    jpg_a = Path("a.jpg")
    dir_a.joinpath(jpg_a).touch()
    jpg_b = Path("b.jpg")
    dir_b.joinpath(jpg_b).touch()
    jpg_c = Path("c.jpg")
    dir_c.joinpath(jpg_c).touch()
    # Create symbolic links
    sym_b = dir_a.joinpath("b.jpg")
    sym_b.symlink_to(jpg_b)
    sym_c = dir_a.joinpath("c")
    sym_c.symlink_to(dir_c, target_is_directory=True)

    # Default follows symbolic links
    paths = list(find_relative_paths(dir_a, [".jpg"]))
    assert {jpg_a, jpg_b, Path("c", "c.jpg")} == set(paths)

    # Do not follow symbolic links
    paths = list(find_relative_paths(dir_a, [".jpg"], follow_symlinks=False))
    assert {jpg_a, jpg_b} == set(paths)


def test_find_relative_paths_group_by_dir(tmp_path):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    vol1_dir = ocr_dir / "vol1"
    vol2_dir = ocr_dir / "vol2"
    for vol_dir in [vol1_dir, vol2_dir]:
        vol_dir.mkdir(exist_ok=True)
        for i in range(4):
            # a text file to include
            (vol_dir / f"{i}.txt").touch()
            # an image file to ignore
            (vol_dir / f"{i}.jpg").touch()
    other_dir = ocr_dir / "no-text-files"
    other_dir.mkdir()

    dir_paths = find_relative_paths(ocr_dir, [".txt"], group_by_dir=True)
    assert isinstance(dir_paths, GeneratorType)
    # cast generator of dir path, files as  dictionary for inspection
    dir_paths = dict(dir_paths)

    # first yielded item (or dictionary key) should be dir as path
    included_dirs = list(dir_paths.keys())
    assert all(isinstance(dirpath, Path) for dirpath in included_dirs)
    # should include relative path versions of directories with text files
    relative_vol1 = vol1_dir.relative_to(ocr_dir)
    assert relative_vol1 in included_dirs
    relative_vol2 = vol2_dir.relative_to(ocr_dir)
    assert relative_vol2 in included_dirs
    # should not include directories without text files
    assert other_dir.relative_to(ocr_dir) not in included_dirs

    # for each volume dir, yielded items should be a list of relative paths
    assert isinstance(dir_paths[relative_vol1], list)
    assert all(isinstance(file, Path) for file in dir_paths[relative_vol1])
    # we expect four files in both groups
    assert len(dir_paths[relative_vol1]) == 4
    assert len(dir_paths[relative_vol2]) == 4
    # spot-check relative path expected to be included
    assert relative_vol1 / "0.txt" in dir_paths[relative_vol1]
    assert relative_vol1 / "3.txt" in dir_paths[relative_vol1]
    assert relative_vol2 / "1.txt" in dir_paths[relative_vol2]
    assert relative_vol2 / "2.txt" in dir_paths[relative_vol2]
