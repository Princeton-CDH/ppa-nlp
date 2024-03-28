import json
from unittest.mock import patch

import pytest

from corppa.utils.filter import filter_pages, save_filtered_corpus

# minimal/mock page data fixture for testing
fixture_page_data = [
    {"work_id": "foo", "label": "i"},
    {"work_id": "bar-p1", "label": "1"},
    {"work_id": "bar-p1", "label": "2"},
    {"work_id": "bar-p1", "label": "3"},
    {"work_id": "baz", "label": "23"},
]


@pytest.fixture
def corpus_file(tmpdir):
    """pytest fixture; creates a jsonl file with fixture_page_data in a tmpdir;
    returns the path object for the jsonl file."""
    corpusfile = tmpdir.join("ppa_pages.jsonl")
    corpusfile.write("\n".join([json.dumps(p) for p in fixture_page_data]))
    return corpusfile


def test_filter_pages(corpus_file):
    source_ids = ["foo", "bar"]
    # use list to consume the generator
    results = list(filter_pages(str(corpus_file), source_ids, disable_progress=True))
    assert len(results) == 4
    assert set([r["work_id"].split("-")[0] for r in results]) == set(source_ids)


@patch("corppa.utils.filter.tqdm")
@patch("corppa.utils.filter.orjsonl")
def test_filter_pages_progressbar(mock_orjsonl, mock_tqdm, corpus_file):
    # test progressbar handling
    # configure mock tqdm iterator to return fixture page data
    mock_tqdm.return_value.__iter__.return_value = fixture_page_data
    # use list to consume the generator
    list(filter_pages(str(corpus_file), ["foo"]))
    mock_orjsonl.stream.assert_called_with(str(corpus_file))
    mock_tqdm.assert_called_with(
        mock_orjsonl.stream.return_value,
        desc="Filtering",
        bar_format="{desc}: {n:,} pages{postfix} | elapsed: {elapsed}",
        disable=False,
    )
    mock_tqdm.return_value.set_postfix_str.assert_any_call("selected 1")


@patch("corppa.utils.filter.filter_pages")
@patch("corppa.utils.filter.orjsonl")
def test_save_filtered_corpus(mock_orjsonl, mock_filter_pages, tmpdir):
    idfile = tmpdir.join("ids.txt")
    ids = ["one", "two", "three", "four"]
    idfile.write("\n".join(ids))
    input_filename = "input.jsonl"
    output_filename = "output.jsonl"

    save_filtered_corpus(input_filename, output_filename, str(idfile))
    # filter should be called with input file and list of ids from text file
    mock_filter_pages.assert_called_with(input_filename, ids)
    # should save result to specified output filename
    mock_orjsonl.save.assert_called_with(
        output_filename, mock_filter_pages.return_value
    )
