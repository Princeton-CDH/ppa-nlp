import pathlib
from unittest.mock import patch

import pytest

from corppa.utils.path_utils import (
    decode_htid,
    encode_htid,
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
