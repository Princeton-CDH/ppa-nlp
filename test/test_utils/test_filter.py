import json
import os
import pathlib
from unittest.mock import patch

import pytest

from corppa.utils.filter import filter_pages, main, save_filtered_corpus

# minimal/mock page data fixture for testing
fixture_page_data = [
    {"work_id": "foo", "label": "i", "order": 2},
    {"work_id": "bar-p1", "label": "1", "order": 1},
    {"work_id": "bar-p1", "label": "2", "order": 2},
    {"work_id": "bar-p1", "label": "3", "order": 3},
    {"work_id": "baz", "label": "23", "order": 27},
]


@pytest.fixture
def corpus_file(tmp_path):
    """pytest fixture; creates a jsonl file with fixture_page_data in a temp dir;
    returns the path object for the jsonl file."""
    corpusfile = tmp_path.joinpath("ppa_pages.jsonl")
    corpusfile.write_text("\n".join([json.dumps(p) for p in fixture_page_data]))
    return corpusfile


def test_filter_work_ids(corpus_file):
    # "bar" corresponds to a source_id not a work_id (since bar-p1 is an excerpt)
    work_ids = ["foo", "bar"]
    results = list(filter_pages(corpus_file, work_ids=work_ids, disable_progress=True))
    assert len(results) == 1
    assert results[0]["work_id"] == "foo"

    work_ids = ["foo", "bar-p1"]
    results = list(filter_pages(corpus_file, work_ids=work_ids, disable_progress=True))
    assert len(results) == 4
    assert set(r["work_id"] for r in results) == set(work_ids)


def test_filter_work_pages(corpus_file):
    work_pages = {"foo": {2}, "bar": {1}, "bar-p1": {3, 5}, "foo": {2}, "baz": {1}}
    results = list(
        filter_pages(
            corpus_file,
            work_pages=work_pages,
            disable_progress=True,
        )
    )
    assert len(results) == 2
    assert set([r["work_id"] for r in results]) == {"foo", "bar-p1"}
    assert set([r["order"] for r in results]) == {2, 3}


def test_filter_include(corpus_file):
    results = list(
        filter_pages(
            corpus_file,
            include_filter={"work_id": "bar-p1", "label": "23"},
            disable_progress=True,
        )
    )
    assert len(results) == 4
    assert set([r["work_id"] for r in results]) == {"bar-p1", "baz"}
    assert set([r["label"] for r in results]) == {"1", "2", "3", "23"}


def test_filter_exclude(corpus_file):
    results = list(
        filter_pages(
            corpus_file,
            exclude_filter={"work_id": "bar-p1", "label": "23"},
            disable_progress=True,
        )
    )
    assert len(results) == 1
    assert set([r["work_id"] for r in results]) == {"foo"}
    assert set([r["label"] for r in results]) == {"i"}


def test_filter_id_and_include(corpus_file):
    # work ids and include filter used in combination
    results = list(
        filter_pages(
            corpus_file,
            work_ids=["bar-p1"],
            include_filter={"label": "2", "work_id": "baz"},
            disable_progress=True,
        )
    )
    assert len(results) == 1
    assert results[0]["work_id"] == "bar-p1"
    assert results[0]["label"] == "2"


def test_filter_id_and_work_pages(corpus_file):
    # provide work ids as well as work pages
    results = list(
        filter_pages(
            corpus_file,
            work_ids=["foo"],
            work_pages={"bar-p1": {2}},
            disable_progress=True,
        )
    )
    assert len(results) == 2
    assert set([r["work_id"] for r in results]) == {"foo", "bar-p1"}
    assert set([r["order"] for r in results]) == {2}


def test_filter_required_args(corpus_file):
    with pytest.raises(ValueError, match="At least one filter must be specified"):
        list(filter_pages(corpus_file))


@patch("corppa.utils.filter.tqdm")
@patch("corppa.utils.filter.orjsonl")
def test_filter_pages_progressbar(mock_orjsonl, mock_tqdm, corpus_file):
    # test progressbar handling
    # configure mock tqdm iterator to return fixture page data
    mock_tqdm.return_value.__iter__.return_value = fixture_page_data
    # use list to consume the generator
    list(filter_pages(corpus_file, ["foo"]))
    mock_orjsonl.stream.assert_called_with(corpus_file)
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
    list(filter_pages(corpus_file, ["foo"], disable_progress=True))
    mock_orjsonl.stream.assert_called_with(corpus_file)
    mock_tqdm.assert_called_with(
        mock_orjsonl.stream.return_value,
        desc="Filtering",
        bar_format="{desc}: checked {n:,} pages{postfix} | elapsed: {elapsed}",
        disable=True,
    )


@patch("corppa.utils.filter.filter_pages")
@patch("corppa.utils.filter.orjsonl")
def test_save_filtered_corpus(mock_orjsonl, mock_filter_pages, tmp_path):
    idfile = tmp_path.joinpath("ids.txt")
    ids = ["one", "two", "three", "four"]
    idfile.write_text("\n".join(ids))
    input_filename = "input.jsonl"
    output_filename = "output.jsonl"

    save_filtered_corpus(input_filename, output_filename, idfile)
    # filter should be called with input file and list of ids from text file
    mock_filter_pages.assert_called_with(
        input_filename,
        work_ids=ids,
        work_pages=None,
        include_filter=None,
        exclude_filter=None,
        disable_progress=False,
    )
    # should save result to specified output filename
    mock_orjsonl.save.assert_called_with(
        output_filename, mock_filter_pages.return_value
    )


def test_save_filtered_corpus_required_args():
    with pytest.raises(ValueError, match="At least one filter must be specified"):
        save_filtered_corpus("pages.jsonl", "filtered.jsonl")


def test_save_filtered_corpus_pgfile_fieldnames(tmp_path):
    pgfile = tmp_path.joinpath("pages.csv")
    pgfile.write_text("work,pg_id\n")
    pgfile.write_text("foo,1\n")
    pgfile.write_text("bar,2\n")

    with pytest.raises(
        ValueError,
        match=f'pgfile {pgfile} must include fields "work_id" and "page_num"',
    ):
        save_filtered_corpus("pages.jsonl", "filtered.jsonl", pgfile=pgfile)


@pytest.mark.parametrize(
    "cli_args, call_params",
    [
        # all required params, default progressbar behavior
        (
            ["filter.py", "pages.json", "subset.jsonl", "--idfile", "id.txt"],
            (
                (pathlib.Path("pages.json"), pathlib.Path("subset.jsonl")),
                {
                    "idfile": pathlib.Path("id.txt"),
                    "pgfile": None,
                    "include_filter": None,
                    "exclude_filter": None,
                    "disable_progress": False,
                },
            ),
        ),
        # disable progress bar
        (
            [
                "filter.py",
                "pages.json.bz2",
                "subset.jsonl.gz",
                "--idfile",
                "id.txt",
                "--no-progress",
            ],
            (
                (pathlib.Path("pages.json.bz2"), pathlib.Path("subset.jsonl.gz")),
                {
                    "idfile": pathlib.Path("id.txt"),
                    "pgfile": None,
                    "include_filter": None,
                    "exclude_filter": None,
                    "disable_progress": True,
                },
            ),
        ),
        # no extension on output file; should add jsonl
        (
            ["filter.py", "pages.json", "subset", "--idfile", "id.txt"],
            (
                (pathlib.Path("pages.json"), pathlib.Path("subset.jsonl")),
                {
                    "idfile": pathlib.Path("id.txt"),
                    "pgfile": None,
                    "include_filter": None,
                    "exclude_filter": None,
                    "disable_progress": False,
                },
            ),
        ),
        # include filter
        (
            ["filter.py", "pages.json", "subset", "--include", "tag=one", "page=2"],
            (
                (pathlib.Path("pages.json"), pathlib.Path("subset.jsonl")),
                {
                    "idfile": None,
                    "pgfile": None,
                    "include_filter": {"tag": "one", "page": "2"},
                    "exclude_filter": None,
                    "disable_progress": False,
                },
            ),
        ),
        # exclude filter
        (
            ["filter.py", "pages.json", "subset", "--exclude", "contains_poetry=Yes"],
            (
                (pathlib.Path("pages.json"), pathlib.Path("subset.jsonl")),
                {
                    "idfile": None,
                    "pgfile": None,
                    "include_filter": None,
                    "exclude_filter": {"contains_poetry": "Yes"},
                    "disable_progress": False,
                },
            ),
        ),
        # pgfile filter
        (
            ["filter.py", "pages.json", "subset", "--pgfile", "pages.csv"],
            (
                (pathlib.Path("pages.json"), pathlib.Path("subset.jsonl")),
                {
                    "idfile": None,
                    "pgfile": pathlib.Path("pages.csv"),
                    "include_filter": None,
                    "exclude_filter": None,
                    "disable_progress": False,
                },
            ),
        ),
    ],
)
@patch("corppa.utils.filter.save_filtered_corpus")
def test_main(mock_save_filtered_corpus, cli_args, call_params, tmp_path):
    # change to temp directory, make sure id file exists and is non-zero
    os.chdir(tmp_path)
    # create an idfile at expected path; arg comes immediately after --idfile
    if "--idfile" in cli_args:
        idfile = tmp_path / cli_args[cli_args.index("--idfile") + 1]
        idfile.write_text("id1\nid2")
    # cerate a csvfile at expected path; args comes immediately after --pgfile
    if "--pgfile" in cli_args:
        pgfile = tmp_path / cli_args[cli_args.index("--pgfile") + 1]
        pgfile.write_text("src_id1,1\nsrc_id2,2")

    # patch in test args for argparse to parse
    with patch("sys.argv", cli_args):
        main()
        args, kwargs = call_params
        mock_save_filtered_corpus.assert_called_with(*args, **kwargs)


def test_main_argparse_error(capsys):
    # call with required parameters but no filters
    with patch("sys.argv", ["filter.py", "pages.json", "subset"]):
        # at least one filter is required
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "At least one filter option must be specified" in captured.err


@patch("corppa.utils.filter.save_filtered_corpus")
def test_main_cleanup(mock_save_filtered_corpus, tmp_path, capsys):
    input_file = tmp_path / "pages.json"
    idfile = tmp_path / "id.txt"
    output_file = tmp_path / "subset.jsonl"

    cli_args = ["filter.py", str(input_file), str(output_file), "--idfile", str(idfile)]

    # change to temp directory, make sure id file exists and is non-zero
    os.chdir(tmp_path)
    # create an idfile at expected path
    idfile = tmp_path / cli_args[cli_args.index("--idfile") + 1]
    idfile.write_text("id1\nid2")

    # as a mock side effect, create a zero size file to be cleaned up
    def create_output(*args, **kwargs):
        output_file.write_text("")

    mock_save_filtered_corpus.side_effect = create_output

    # patch in arguments for arg.parse to load
    with patch("sys.argv", cli_args):
        main()
        assert not output_file.exists()
        captured = capsys.readouterr()
        assert "No pages were selected, removing empty output file" in captured.out

    # with cleanup disabled, zero-size file should not be removed
    cli_args.append("--no-cleanup")
    with patch("sys.argv", cli_args):
        main()
        assert output_file.exists()
        captured = capsys.readouterr()
        # should still report on the empty file
        assert "No pages were selected" in captured.out


@patch("corppa.utils.filter.save_filtered_corpus")
def test_main_idfile_nonexistent(mock_save_filtered_corpus, capsys):
    with patch(
        "sys.argv", ["f.py", "foo.jsonl", "out.jsonl", "--idfile", "/not/a/real/id.txt"]
    ):
        with pytest.raises(SystemExit) as execinfo:
            main()
        assert execinfo.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


@patch("corppa.utils.filter.save_filtered_corpus")
def test_main_idfile_empty(mock_save_filtered_corpus, capsys, tmp_path):
    idfile = tmp_path / "id.txt"
    idfile.touch()
    with patch("sys.argv", ["f.py", "foo.jsonl", "out.jsonl", "--idfile", str(idfile)]):
        with pytest.raises(SystemExit) as execinfo:
            main()
        assert execinfo.value.code == 1
    captured = capsys.readouterr()
    assert "is zero size" in captured.err


@patch("corppa.utils.filter.save_filtered_corpus")
def test_main_pgfile_empty(mock_save_filtered_corpus, capsys, tmp_path):
    pgfile = tmp_path / "pages.csv"
    pgfile.touch()
    with patch("sys.argv", ["f.py", "foo.jsonl", "out.jsonl", "--pgfile", str(pgfile)]):
        with pytest.raises(SystemExit) as execinfo:
            main()
        assert execinfo.value.code == 1
    captured = capsys.readouterr()
    assert "is zero size" in captured.err


@patch("corppa.utils.filter.save_filtered_corpus")
def test_main_outfile_exists(mock_save_filtered_corpus, capsys, tmp_path):
    idfile = tmp_path / "id.txt"
    idfile.write_text("id1\nid2")
    outfile = tmp_path / "subset.jsonl"
    outfile.touch()
    with patch(
        "sys.argv", ["f.py", "foo.jsonl", str(outfile), "--idfile", str(idfile)]
    ):
        with pytest.raises(SystemExit) as execinfo:
            main()
        assert execinfo.value.code == 1
    captured = capsys.readouterr()
    assert "already exists" in captured.err
