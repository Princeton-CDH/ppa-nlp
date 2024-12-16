import sys
from inspect import isgenerator
from unittest.mock import MagicMock, call, patch

import pytest

from corppa.poetry_detection.annotation.process_adjudication_data import (
    get_excerpt_entries,
    get_excerpts,
    process_adjudication_data,
    process_page_annotation,
)


def test_get_excerpts():
    page_annotation = {"text": "some page text"}

    # Missing spans field
    with pytest.raises(ValueError, match="Page annotation missing 'spans' field"):
        get_excerpts(page_annotation)

    # Empty spans field
    page_annotation["spans"] = []
    assert get_excerpts(page_annotation) == []

    # Regular case (i.e. non-empty spans field)
    page_annotation["spans"].append({"start": 0, "end": 4})
    page_annotation["spans"].append({"start": 10, "end": 14})
    results = get_excerpts(page_annotation)
    assert results[0] == {"start": 0, "end": 4, "text": "some"}
    assert results[1] == {"start": 10, "end": 14, "text": "text"}

    # Missing text field
    blank_page = {"spans": []}
    assert get_excerpts(blank_page) == []


@patch("corppa.poetry_detection.annotation.process_adjudication_data.get_excerpts")
def test_process_page_annotation(mock_get_excerpts):
    mock_get_excerpts.return_value = ["some", "poetry", "excerpts"]
    page_annotation = {
        "id": "some-page-id",
        "work_id": "some-work-id",
        "meta": {"title": "some-title", "author": "some-author", "year": "some-year"},
        "spans": "some-spans",
    }
    result = process_page_annotation(page_annotation)
    assert result == {
        "page_id": "some-page-id",
        "work_id": "some-work-id",
        "work_title": "some-title",
        "work_author": "some-author",
        "work_year": "some-year",
        "excerpts": ["some", "poetry", "excerpts"],
        "n_excerpts": 3,
    }
    mock_get_excerpts.assert_called_once_with(page_annotation)


def test_get_excerpt_entries():
    page_meta = {
        "page_id": "some-page-id",
        "work_id": "some-work-id",
        "work_title": "some-title",
        "work_author": "some-author",
        "work_year": "some-year",
    }
    excerpts = [
        {"start": 0, "end": 3, "text": "a"},
        {"start": 5, "end": 6, "text": "b"},
    ]
    page_data = page_meta | {"excerpts": excerpts}
    expected_results = [page_meta | excerpt for excerpt in excerpts]

    result = get_excerpt_entries(page_data)
    assert isgenerator(result)
    assert list(result) == expected_results


@patch(
    "corppa.poetry_detection.annotation.process_adjudication_data.get_excerpt_entries"
)
@patch(
    "corppa.poetry_detection.annotation.process_adjudication_data.process_page_annotation"
)
@patch("corppa.poetry_detection.annotation.process_adjudication_data.orjsonl")
@patch("corppa.poetry_detection.annotation.process_adjudication_data.tqdm")
def test_process_adjudication_data(
    mock_tqdm,
    mock_orjsonl,
    mock_process_page_annotation,
    mock_get_excerpt_entries,
    tmpdir,
):
    input_jsonl = tmpdir / "input.jsonl"
    input_jsonl.write_text("some\ntext\n", encoding="utf-8")
    out_excerpts = tmpdir / "output.csv"

    # Default
    csv_fields = [
        "page_id",
        "work_id",
        "work_title",
        "work_author",
        "work_year",
        "start",
        "end",
        "text",
    ]
    mock_orjsonl.stream.return_value = "jsonl stream"
    mock_tqdm.return_value = ["a", "b"]
    mock_process_page_annotation.side_effect = lambda x: f"page {x}"
    mock_get_excerpt_entries.return_value = [{k: "test" for k in csv_fields}]

    process_adjudication_data(input_jsonl, "out.jsonl", out_excerpts)
    mock_orjsonl.stream.assert_called_once_with(input_jsonl)
    mock_tqdm.assert_called_once_with("jsonl stream", total=2, disable=False)
    assert mock_process_page_annotation.call_count == 2
    mock_process_page_annotation.assert_has_calls([call("a"), call("b")])
    assert mock_orjsonl.append.call_count == 2
    mock_orjsonl.append.assert_has_calls(
        [call("out.jsonl", "page a"), call("out.jsonl", "page b")]
    )
    assert mock_get_excerpt_entries.call_count == 2
    mock_get_excerpt_entries.assert_has_calls([call("page a"), call("page b")])
    csv_text = ",".join(csv_fields) + "\n"
    csv_text += ",".join(["test"] * 8) + "\n"
    csv_text += ",".join(["test"] * 8) + "\n"
    assert out_excerpts.read_text(encoding="utf-8") == csv_text

    # Disable progress
    mock_orjsonl.reset_mock()
    mock_orjsonl.stream.return_value = "jsonl stream"
    mock_tqdm.reset_mock()
    mock_tqdm.return_value = ["a", "b"]
    process_adjudication_data(
        input_jsonl, "out.jsonl", out_excerpts, disable_progress=True
    )
    mock_orjsonl.stream.assert_called_once_with(input_jsonl)
    mock_tqdm.assert_called_once_with("jsonl stream", total=2, disable=True)
