# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import call, patch

import pytest

from corppa.poetry_detection.core import LabeledExcerpt
from corppa.poetry_detection.passim.get_passim_results import (
    add_excerpts,
    build_passim_excerpt,
    extract_passim_spans,
    get_page_texts,
    get_passim_span,
)


@patch("corppa.poetry_detection.passim.get_passim_results.orjsonl")
def test_get_page_texts(mock_orjsonl):
    # Setup corpus data
    jsonl_data = [
        {"id": "a", "text": "1"},
        {"id": "b", "text": "2"},
        {"id": "c", "text": "3"},
    ]
    mock_orjsonl.stream.return_value = jsonl_data

    expected_results = {"a": "1", "c": "3"}
    assert get_page_texts({"a", "c", "z"}, "input jsonl") == expected_results
    mock_orjsonl.stream.assert_called_once_with("input jsonl")


@patch.object(LabeledExcerpt, "correct_page_excerpt")
def test_build_passim_excerpt(mock_correct_excerpt):
    span_record = {
        "poem_id": "poem_a",
        "ref_corpus": "ref",
        "ref_start": 11,
        "ref_end": 12,
        "ref_excerpt": "B",
        "page_start": 1,
        "page_end": 2,
        "ppa_excerpt": "b",
        "matches": 1,
    }
    # Basic case (without correction)
    expected_result = LabeledExcerpt(
        page_id="page_id",
        ppa_span_start=1,
        ppa_span_end=2,
        ppa_span_text="b",
        poem_id="poem_a",
        ref_corpus="ref",
        ref_span_start=11,
        ref_span_end=12,
        ref_span_text="B",
        detection_methods={"passim"},
        identification_methods={"passim"},
        notes="passim: 1 char matches",
    )
    assert build_passim_excerpt("page_id", span_record) == expected_result
    mock_correct_excerpt.assert_not_called()
    # With correction
    mock_correct_excerpt.side_effect = "a"
    assert build_passim_excerpt("page_id", span_record, ppa_page_text="ppa_page") == "a"
    mock_correct_excerpt.assert_called_once_with("ppa_page")


def test_get_passim_span():
    passim_record = {
        "id": "poem_id",
        "corpus": "ref",
        "begin": "ref_start_idx",
        "end": "ref_end_idx",
        "id2": "ppa_page_id",
        "begin2": "page_start_idx",
        "end2": "page_end_idx",
        "matches": "# char matches",
        "s1": "Aligned text\nwith dashes",
        "s2": "Aligned text with\t------",
    }
    assert get_passim_span(passim_record) == {
        "page_id": "ppa_page_id",
        "page_start": "page_start_idx",
        "page_end": "page_end_idx",
        "poem_id": "poem_id",
        "ref_corpus": "ref",
        "ref_start": "ref_start_idx",
        "ref_end": "ref_end_idx",
        "matches": "# char matches",
        "aligned_ref_excerpt": "Aligned text with dashes",
        "aligned_ppa_excerpt": "Aligned text with ------",
    }


@patch("corppa.poetry_detection.passim.get_passim_results.get_passim_span")
@patch("corppa.poetry_detection.passim.get_passim_results.orjsonl")
def test_extract_passim_spans(mock_orjsonl, mock_get_span, tmp_path):
    # Build test input directory
    ## Create top-level passim directory
    passim_dir = tmp_path / "passim_output"
    passim_dir.mkdir()
    ## Create align directory
    align_dir = passim_dir / "align.json"
    align_dir.mkdir()
    ## Create some input files
    input_a = align_dir / "a.json"
    input_a.touch()
    input_b = align_dir / "b.json"
    input_b.touch()
    ## An ignored file
    ignore_c = align_dir / "c.txt"
    ignore_c.touch()

    # Error on bad input directory
    error_message = (
        f"Error: Alignment directory '{align_dir}/align.json' does not exist"
    )
    with pytest.raises(ValueError, match=error_message):
        _ = list(extract_passim_spans(align_dir, disable_progress=True))
    mock_orjsonl.assert_not_called()
    mock_get_span.assert_not_called()

    # Typical case
    mock_orjsonl.stream.side_effect = [iter(["a1"]), iter(["b1", "b2"])]
    mock_get_span.side_effect = ["r1", "r2", "r3"]

    results = list(extract_passim_spans(passim_dir, disable_progress=True))
    assert results == ["r1", "r2", "r3"]
    assert mock_orjsonl.stream.call_count == 2
    # NOTE: passes locally in this order but fails on GitHub due to ordering
    mock_orjsonl.stream.assert_has_calls([call(input_a), call(input_b)], any_order=True)
    assert mock_get_span.call_count == 3
    mock_get_span.assert_has_calls([call("a1"), call("b1"), call("b2")])


@patch("corppa.poetry_detection.passim.get_passim_results.orjsonl")
def test_add_excerpts(mock_orjsonl):
    page_results = {
        # Single span
        "a": {
            "poem_spans": [
                {
                    "page_start": 1,
                    "page_end": 3,
                    "poem_id": "poem_a",
                    "ref_corpus": "r1",
                    "ref_start": 11,
                    "ref_end": 13,
                }
            ]
        },
        # Two spans
        "b": {
            "poem_spans": [
                {
                    "page_start": 2,
                    "page_end": 5,
                    "poem_id": "poem_1",
                    "ref_corpus": "r2",
                    "ref_start": 3,
                    "ref_end": 6,
                },
                {
                    "page_start": 6,
                    "page_end": 11,
                    "poem_id": "poem_b",
                    "ref_corpus": "r1",
                    "ref_start": 0,
                    "ref_end": 6,
                },
            ],
        },
        # No spans
        "page_c": {"poem_spans": []},
    }

    ppa_corpus = [{"id": "a", "text": "text"}, {"id": "b", "text": "hello world!"}]
    r1_corpus = [
        {"id": "poem_a", "corpus": "r1", "text": "abcdefghijklmnop"},
        {"id": "poem_b", "corpus": "r1", "text": "poetry"},
        {"id": "poem_c", "corpus": "r1", "text": "ignored"},
    ]
    r2_corpus = [
        {"id": "poem_0", "corpus": "r2", "text": "skip"},
        {"id": "poem_1", "corpus": "r2", "text": "some text"},
    ]

    mock_orjsonl.stream.side_effect = [
        iter(ppa_corpus),
        iter(r1_corpus),
        iter(r2_corpus),
    ]

    add_excerpts(
        page_results,
        "ppa_passim_corpus",
        ["r1_corpus", "r2_corpus"],
        disable_progress=True,
    )
    assert mock_orjsonl.stream.call_count == 3
    mock_orjsonl.stream.assert_has_calls(
        [call("ppa_passim_corpus"), call("r1_corpus"), call("r2_corpus")]
    )

    # Note: this function modifies the input page_results
    expected_results = {
        "a": {
            "poem_spans": [
                {
                    "page_start": 1,
                    "page_end": 3,
                    "ppa_excerpt": "ex",
                    "poem_id": "poem_a",
                    "ref_corpus": "r1",
                    "ref_start": 11,
                    "ref_end": 13,
                    "ref_excerpt": "lm",
                }
            ]
        },
        "b": {
            "poem_spans": [
                {
                    "page_start": 2,
                    "page_end": 5,
                    "ppa_excerpt": "llo",
                    "poem_id": "poem_1",
                    "ref_corpus": "r2",
                    "ref_start": 3,
                    "ref_end": 6,
                    "ref_excerpt": "e t",
                },
                {
                    "page_start": 6,
                    "page_end": 11,
                    "ppa_excerpt": "world",
                    "poem_id": "poem_b",
                    "ref_corpus": "r1",
                    "ref_start": 0,
                    "ref_end": 6,
                    "ref_excerpt": "poetry",
                },
            ],
        },
        "page_c": {"poem_spans": []},
    }
    assert page_results == expected_results
