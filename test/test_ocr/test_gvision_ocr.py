import pathlib
from types import GeneratorType
from unittest.mock import patch

import pytest

from corppa.ocr.gvision_ocr import image_relpath_generator, ocr_images


def test_image_relpath_generator(tmp_path):
    jpg_a = pathlib.Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    txt_b = pathlib.Path("b.txt")
    tmp_path.joinpath(txt_b).touch()

    # I. Single ext
    paths = image_relpath_generator(tmp_path, [".jpg"])
    assert isinstance(paths, GeneratorType)
    assert [jpg_a] == list(paths)

    # II. Multiple exts
    tif_c = pathlib.Path("c.tif")
    tmp_path.joinpath(tif_c).touch()
    paths = list(image_relpath_generator(tmp_path, [".jpg", ".tif"]))
    assert {jpg_a, tif_c} == set(paths)

    # III. Extension handling is case insensitive
    jpg_d = pathlib.Path("d.JPG")
    tmp_path.joinpath(jpg_d).touch()
    paths_a = list(image_relpath_generator(tmp_path, [".jpg"]))
    paths_b = list(image_relpath_generator(tmp_path, [".JPG"]))
    assert set(paths_a) == set(paths_b)
    assert {jpg_a, jpg_d} == set(paths_a)


def test_image_relpath_generator_nested(tmp_path):
    img_dir = tmp_path.joinpath(tmp_path, "images")
    img_dir.mkdir()
    jpg_a = pathlib.Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    jpg_b = pathlib.Path("b.jpg")
    img_dir.joinpath(jpg_b).touch()

    paths = image_relpath_generator(img_dir, [".jpg"])
    assert {jpg_b} == set(paths)

    paths = image_relpath_generator(tmp_path, [".jpg"])
    assert {jpg_a, pathlib.Path("images", "b.jpg")} == set(paths)


def test_image_relpath_hidden_dirs(tmp_path):
    jpg_a = pathlib.Path("a.jpg")
    tmp_path.joinpath(jpg_a).touch()
    hidden_dir = tmp_path.joinpath(".hidden")
    hidden_dir.mkdir()
    jpg_b = pathlib.Path("b.jpg")
    hidden_dir.joinpath(jpg_b).touch()

    paths = list(image_relpath_generator(tmp_path, [".jpg"]))
    assert [jpg_a] == paths


def test_image_relpath_generator_symbolic_links(tmp_path):
    """
    Test directory sturcture:
     dir_a:
        a.jpg
        b.jpg (symbolic link, file: dir_b/b.jpg)
        dir_c (symbolic link, dir: dir_b/dir_c)
     dir_b:
        b.jpg
        dir_c:
          c.jpg
    """
    # Create directories
    dir_a = tmp_path.joinpath("dir_a")
    dir_a.mkdir()
    dir_b = tmp_path.joinpath("dir_b")
    dir_b.mkdir()
    dir_c = dir_b.joinpath("dir_c")
    dir_c.mkdir()
    # Create files
    jpg_a = pathlib.Path("a.jpg")
    dir_a.joinpath(jpg_a).touch()
    jpg_b = pathlib.Path("b.jpg")
    dir_b.joinpath(jpg_b).touch()
    jpg_c = pathlib.Path("c.jpg")
    dir_c.joinpath(jpg_c).touch()
    # Create symbolic links
    sym_b = dir_a.joinpath("b.jpg")
    sym_b.symlink_to(jpg_b)
    sym_c = dir_a.joinpath("c")
    sym_c.symlink_to(dir_c, target_is_directory=True)

    # Default follows symbolic links
    paths = list(image_relpath_generator(dir_a, [".jpg"]))
    assert {jpg_a, jpg_b, pathlib.Path("c", "c.jpg")} == set(paths)

    # Do not follow symbolic links
    paths = list(image_relpath_generator(dir_a, [".jpg"], follow_symlinks=False))
    assert {jpg_a, jpg_b} == set(paths)


@patch("corppa.ocr.gvision_ocr.google_vision", None)
def test_ocr_images_no_gvision(capsys):
    with pytest.raises(SystemExit):
        ocr_images(pathlib.Path("in"), pathlib.Path("out"), set())
    captured = capsys.readouterr()
    assert "does not contain google-cloud-vision" in captured.err
