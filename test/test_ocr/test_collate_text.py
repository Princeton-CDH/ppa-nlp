import json
import pathlib

from corppa.ocr.collate_txt import collate_txt, page_number


def test_page_number():
    assert page_number(pathlib.Path("CW0112029406_00180.txt")) == "0018"


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
    assert "Created JSON file for 1 directory with 3 total text files." in captured.out
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
