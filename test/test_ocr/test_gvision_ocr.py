from pathlib import Path
from types import GeneratorType
from unittest.mock import call, patch

import pytest

from corppa.ocr.gvision_ocr import (
    ocr_image_via_gvision,
    ocr_images,
)


@patch("corppa.ocr.gvision_ocr.google_vision", None)
def test_ocr_image_via_gvision_no_gvision(capsys):
    with pytest.raises(SystemExit):
        ocr_image_via_gvision(None, Path("in.jpg"), Path("out.txt"), Path("out.json"))
    captured = capsys.readouterr()
    assert "does not contain google-cloud-vision" in captured.err


@patch("corppa.ocr.gvision_ocr.google_vision", None)
def test_ocr_images_no_gvision(capsys):
    with pytest.raises(SystemExit):
        ocr_images(Path("in"), Path("out"), set())
    captured = capsys.readouterr()
    assert "does not contain google-cloud-vision" in captured.err


@patch("corppa.ocr.gvision_ocr.find_relative_paths")
@patch("corppa.ocr.gvision_ocr.ocr_image_via_gvision")
@patch("corppa.ocr.gvision_ocr.google_vision")
def test_ocr_images(
    mock_gvision, mock_ocr_image, mock_find_relative_paths, tmp_path, capsys
):
    # Setup up mock clientp
    mock_client = mock_gvision.ImageAnnotatorClient
    img_dir = tmp_path.joinpath("images")
    img_dir.mkdir()
    ocr_dir = tmp_path.joinpath("ocr")
    ocr_dir.mkdir()
    # Create output ocr for b, so b.jpg will be skipped
    ocr_dir.joinpath("b.txt").touch()

    mock_client.return_value = "client_placeholder"
    mock_find_relative_paths.return_value = [
        Path("a.jpg"),
        Path("b.jpg"),
        Path("subdir", "c.jpg"),
    ]

    reporting = ocr_images(img_dir, ocr_dir, [".jpg"])
    assert mock_client.call_count == 1
    # Check that subdirectory for c was created
    assert ocr_dir.joinpath("subdir").is_dir()
    # Check ocr calls
    assert mock_ocr_image.call_count == 2
    calls = [
        call(
            "client_placeholder",
            img_dir.joinpath("a.jpg"),
            ocr_dir.joinpath("a.txt"),
            ocr_dir.joinpath("a.json"),
        ),
        call(
            "client_placeholder",
            img_dir.joinpath("subdir", "c.jpg"),
            ocr_dir.joinpath("subdir", "c.txt"),
            ocr_dir.joinpath("subdir", "c.json"),
        ),
    ]
    mock_ocr_image.assert_has_calls(calls)
    # Check output
    assert {"ocr_count": 2, "skip_count": 1} == reporting

    captured = capsys.readouterr()
    assert "2 images OCR'd & 1 images skipped." in captured.err
