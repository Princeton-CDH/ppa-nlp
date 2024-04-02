import json
import os
from unittest.mock import patch

import pytest

from corppa.utils.filter import filter_pages, save_filtered_corpus, main

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
        bar_format="{desc}: checked {n:,} pages{postfix} | elapsed: {elapsed}",
        disable=False,
    )
    mock_tqdm.return_value.set_postfix_str.assert_any_call("selected 1")


@patch("corppa.utils.filter.tqdm")
@patch("corppa.utils.filter.orjsonl")
def test_filter_pages_noprogressbar(mock_orjsonl, mock_tqdm, corpus_file):
    # test disabling progressbar
    # configure mock tqdm iterator to return fixture page data
    mock_tqdm.return_value.__iter__.return_value = fixture_page_data
    # use list to consume the generator
    list(filter_pages(str(corpus_file), ["foo"], disable_progress=True))
    mock_orjsonl.stream.assert_called_with(str(corpus_file))
    mock_tqdm.assert_called_with(
        mock_orjsonl.stream.return_value,
        desc="Filtering",
        bar_format="{desc}: checked {n:,} pages{postfix} | elapsed: {elapsed}",
        disable=True,
    )


@patch("corppa.utils.filter.filter_pages")
@patch("corppa.utils.filter.orjsonl")
def test_save_filtered_corpus(mock_orjsonl, mock_filter_pages, tmpdir):
    idfile = tmpdir.join("ids.txt")
    ids = ["one", "two", "three", "four"]
    idfile.write("\n".join(ids))
    input_filename = "input.jsonl"
    output_filename = "output.jsonl"

    save_filtered_corpus(input_filename, str(idfile), output_filename)
    # filter should be called with input file and list of ids from text file
    mock_filter_pages.assert_called_with(input_filename, ids, disable_progress=False)
    # should save result to specified output filename
    mock_orjsonl.save.assert_called_with(
        output_filename, mock_filter_pages.return_value
    )


@pytest.mark.parametrize(
    "cli_args, call_params",
    [
        # all required params, default progressbar behavior
        (
            ["filter.py", "pages.json", "id.txt", "subset.jsonl"],
            (
                ("pages.json", "id.txt", "subset.jsonl"),
                {"disable_progress": False},
            ),
        ),
        # disable progress bar
        (
            [
                "filter.py",
                "pages.json.bz2",
                "id.txt",
                "subset.jsonl.gz",
                "--no-progress",
            ],
            (
                ("pages.json.bz2", "id.txt", "subset.jsonl.gz"),
                {"disable_progress": True},
            ),
        ),
        # no extension on output file; should add jsonl
        (
            ["filter.py", "pages.json", "id.txt", "subset"],
            (("pages.json", "id.txt", "subset.jsonl"), {"disable_progress": False}),
        ),
    ],
)
@patch("corppa.utils.filter.save_filtered_corpus")
def test_main(mock_save_filtered_corpus, cli_args, call_params, tmp_path):
    # change to temp directory, make sure id file exists and is non-zero
    os.chdir(tmp_path)
    idfile = tmp_path / cli_args[2]
    idfile.write_text("id1\nid2")

    # patch in test args for argparse to parse
    with patch("sys.argv", cli_args):
        main()
        args, kwargs = call_params
        mock_save_filtered_corpus.assert_called_with(*args, **kwargs)


@patch("corppa.utils.filter.save_filtered_corpus")
@patch("corppa.utils.filter.exit")
def test_main_idfile_nonexistent(mock_exit, mock_save_filtered_corpus, capsys):
    with patch("sys.argv", ["f.py", "foo.jsonl", "/not/a/real/id.txt", "out.jsonl"]):
        main()
    mock_exit.assert_called_with(-1)
    captured = capsys.readouterr()
    assert "does not exist" in captured.out


@patch("corppa.utils.filter.save_filtered_corpus")
@patch("corppa.utils.filter.exit")
def test_main_idfile_empty(mock_exit, mock_save_filtered_corpus, capsys, tmp_path):
    idfile = tmp_path / "id.txt"
    idfile.touch()
    with patch("sys.argv", ["f.py", "foo.jsonl", str(idfile), "out.jsonl"]):
        main()
    mock_exit.assert_called_with(-1)
    captured = capsys.readouterr()
    assert "is empty" in captured.out


@patch("corppa.utils.filter.save_filtered_corpus")
@patch("corppa.utils.filter.exit")
def test_main_outfile_exists(mock_exit, mock_save_filtered_corpus, capsys, tmp_path):
    idfile = tmp_path / "id.txt"
    idfile.write_text("id1\nid2")
    outfile = tmp_path / "subset.jsonl"
    outfile.touch()
    with patch("sys.argv", ["f.py", "foo.jsonl", str(idfile), str(outfile)]):
        main()
    mock_exit.assert_called_with(-1)
    captured = capsys.readouterr()
    assert "already exists" in captured.out
