from inspect import isgenerator
from pathlib import Path
from unittest.mock import call, patch

import pytest

from corppa.utils.build_text_corpus import (
    build_text_corpus,
    get_text_record,
    save_text_corpus,
)


def test_get_text_record(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Some\n text.", encoding="utf-8")

    result = get_text_record(test_file)
    expected_result = {"id": "test", "text": "Some\n text."}
    assert result == expected_result


@patch("corppa.utils.build_text_corpus.get_text_record")
def test_build_text_corpus(mock_get_text_record, tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    txt_a = tmp_path / "a.txt"
    txt_a.touch()
    txt_b = corpus_dir / "b.txt"
    txt_b.touch()
    sub_dir = corpus_dir / "c"
    sub_dir.mkdir()
    txt_c = sub_dir / "c.txt"
    txt_c.write_text("test", encoding="utf-8")
    other_file = sub_dir / "other.xml"
    other_file.touch()

    # Simple case
    mock_get_text_record.return_value = "some record"
    results = build_text_corpus(sub_dir)
    assert isgenerator(results)
    assert list(results) == ["some record"]
    mock_get_text_record.assert_called_once_with(txt_c)

    # Nested directories & ignored files
    mock_get_text_record.reset_mock()
    mock_get_text_record.side_effect = ["b", "c"]
    results = build_text_corpus(corpus_dir)
    assert list(results) == ["b", "c"]
    assert mock_get_text_record.call_count == 2
    mock_get_text_record.assert_has_calls([call(txt_b), call(txt_c)])


@patch("corppa.utils.build_text_corpus.build_text_corpus")
@patch("corppa.utils.build_text_corpus.orjsonl")
def test_save_text_corpus(mock_orjsonl, mock_build_text_corpus):
    mock_build_text_corpus.return_value = "text corpus"
    save_text_corpus("input dir", "output file")
    mock_build_text_corpus.assert_called_once_with("input dir")
    mock_orjsonl.save.assert_called_once_with("output file", "text corpus")
