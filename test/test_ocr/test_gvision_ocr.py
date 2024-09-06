import pathlib
from types import GeneratorType
from unittest.mock import patch

import pytest

from corppa.ocr.gvision_ocr import image_relpath_generator


def test_image_relpath_generator(tmp_path):
    relp_a = pathlib.Path("a.jpg")
    tmp_path.joinpath(relp_a).touch()
    relp_b = pathlib.Path("b.txt")
    tmp_path.joinpath(relp_b).touch()

    paths = image_relpath_generator(tmp_path, {".jpg"})
    assert isinstance(paths, GeneratorType)

    paths = set(paths)
    assert relp_a in paths
    assert relp_b not in paths

    # TODO: add nested checks

    # TODO: test symbolic link parameter!

    # TODO: multiple extensions & variation in case
