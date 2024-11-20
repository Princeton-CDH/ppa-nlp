import json
import pathlib
from unittest.mock import patch

import pytest

from corppa.ocr.collate_txt import collate_txt, main


def test_collate_txt(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_dir_subdir = input_dir / "001"
    input_dir_subdir.mkdir()
    vol_id = "CW012345"
    input_vol = input_dir_subdir / vol_id
    input_vol.mkdir()
    text_contents = {
        f"{vol_id}_00010.txt": "title page",
        f"{vol_id}_00020.txt": "table of contents",
        f"{vol_id}_00030.txt": "an introduction to prosody",
    }
    for filename, text in text_contents.items():
        (input_vol / filename).open("w").write(text)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    collate_txt(input_dir, output_dir)
    captured = capsys.readouterr()
    # check summary output
    assert (
        "Created JSON file for 1 directory with 3 total text files; skipped 0."
        in captured.out
    )
    # check expected directory and file exists
    assert (output_dir / "001").exists()
    expected_jsonfile = (output_dir / "001" / vol_id).with_suffix(".json")
    assert expected_jsonfile.exists()
    with expected_jsonfile.open() as testdatafile:
        data = json.load(testdatafile)
    for page_num in ["0001", "0002", "0003"]:
        # page numbers from filenames should be present as keys
        assert page_num in data
        # text content should be present, accessible by page number
        assert data[page_num] == text_contents[f"{vol_id}_{page_num}0.txt"]

    # running it again should skip, since json file exists
    collate_txt(input_dir, output_dir)
    captured = capsys.readouterr()
    assert "skipped 1" in captured.out


@patch("corppa.ocr.collate_txt.collate_txt")
def test_main(mock_collate_txt, capsys, tmp_path):
    # patch in test args for argparse to parse

    # non-existent input dir
    with patch(
        "sys.argv", ["collate_txt", "non-existent-input", "non-existent-output"]
    ):
        with pytest.raises(SystemExit) as sysexit:
            main()
        assert sysexit.value.code == 1
        captured = capsys.readouterr()
        assert (
            "Error: input directory non-existent-input does not exist" in captured.err
        )
        mock_collate_txt.assert_not_called()

    # valid input dir, non-existent output dir
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # invalid output dir path: exists as a file
    output_dir = tmp_path / "output"
    output_dir.touch()

    with patch("sys.argv", ["collate_txt", str(input_dir), str(output_dir)]):
        with pytest.raises(SystemExit) as sysexit:
            main()
        assert sysexit.value.code == 1

        captured = capsys.readouterr()
        assert f"Error creating output directory {output_dir}" in captured.err
        assert "exists" in captured.err

    # invalid output dir path: parent directory doesn't exist
    output_dir.unlink()
    output_dir = tmp_path / "parent" / "output"

    with patch("sys.argv", ["collate_txt", str(input_dir), str(output_dir)]):
        with pytest.raises(SystemExit) as sysexit:
            main()
        assert sysexit.value.code == 1

        captured = capsys.readouterr()
        assert f"Error creating output directory {output_dir}" in captured.err
        assert "No such file or directory" in captured.err
        mock_collate_txt.assert_not_called()

    # valid output directory; default progress bar (on)
    output_dir = tmp_path / "output"
    with patch("sys.argv", ["collate_txt", str(input_dir), str(output_dir)]):
        main()
        mock_collate_txt.assert_called_with(input_dir, output_dir, show_progress=True)
        captured = capsys.readouterr()
        assert "Creating output directory" in captured.out

    # valid output directory; disable progress bar
    output_dir = tmp_path / "output"
    with patch(
        "sys.argv", ["collate_txt", str(input_dir), str(output_dir), "--no-progress"]
    ):
        main()
        mock_collate_txt.assert_called_with(input_dir, output_dir, show_progress=False)
