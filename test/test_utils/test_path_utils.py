import pathlib
import pytest

from corppa.utils.path_utils import encode_htid, decode_htid
from corppa.utils.path_utils import get_stub_dir, get_vol_dir


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


def test_get_stub_dir():
    # Gale
    assert get_stub_dir("Gale", "CB0127060085") == "100"
    # HathiTrust
    assert get_stub_dir("HathiTrust", "mdp.39015003633594") == "mdp"
    # Other
    with pytest.raises(ValueError, match="Unknown source 'invalid src'"):
        get_stub_dir("invalid src", "xxx0000")


def test_get_vol_dir():
    # Gale
    assert get_vol_dir("Gale", "CB0127060085") == pathlib.Path(
        "Gale", "100", "CB0127060085"
    )
    # HathiTrust
    assert get_vol_dir("HathiTrust", "mdp.39015003633594") == pathlib.Path(
        "HathiTrust", "mdp", "mdp.39015003633594"
    )
    assert get_vol_dir("HathiTrust", "dul1.ark:/13960/t5w67998k") == pathlib.Path(
        "HathiTrust", "dul1", "dul1.ark+=13960=t5w67998k"
    )
    # Other
    with pytest.raises(ValueError, match="Unknown source 'invalid src'"):
        get_vol_dir("invalid src", "xxx0000")
