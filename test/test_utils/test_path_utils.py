import pathlib
import pytest

from unittest.mock import patch
from corppa.utils.path_utils import (
    encode_htid,
    decode_htid,
    get_ppa_source,
    get_stub_dir,
    get_vol_dir,
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


@pytest.mark.parametrize("source", ["Gale", "HathiTrust", "Unknown"])
@patch("corppa.utils.path_utils.get_stub_dir", return_value="stub_name")
@patch("corppa.utils.path_utils.get_ppa_source")
def test_get_vol_dir(mock_get_ppa_source, mock_get_stub_dir, source):
    # Set returned source value
    mock_get_ppa_source.return_value = source
    vol_id = f"{source}_id"
    if source == "Gale":
        assert get_vol_dir(vol_id) == pathlib.Path(source, "stub_name", vol_id)
        mock_get_stub_dir.assert_called_with(source, vol_id)
    elif source == "HathiTrust":
        # TODO: Update once HathiTrust directory conventions are finalized
        with pytest.raises(
            NotImplementedError, match="HathiTrust volume directory conventions TBD"
        ):
            get_vol_dir(vol_id)
        mock_get_stub_dir.assert_not_called()
    else:
        with pytest.raises(ValueError, match=f"Unknown source '{source}'"):
            get_vol_dir(vol_id)
        mock_get_stub_dir.assert_not_called()

    mock_get_ppa_source.assert_called_with(vol_id)
