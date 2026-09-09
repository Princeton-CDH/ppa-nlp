# Copyright (c) 2024-2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import signal
import tarfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import orjsonl
import polars as pl
import pytest
from tqdm import tqdm

import corppa.utils.dataset_prep as dataset_prep
from corppa.utils.dataset_prep import (
    add_zip_file_to_tar,
    align_pages,
    align_shifted_pages,
    find_corpus_file,
    get_ht1930_work_ids,
    get_ht_zipfile_path,
    get_zip_textfiles,
    get_zipfile_pages,
    longest_increasing_subseq,
    main,
    open_ht_zipfile,
    plot_alignment,
    process_gale_work,
    process_ht1930_work,
    process_ht_work,
    process_work,
    review_alignment,
    zip_image_filenames,
)

WORK_ID = "htid:test.12345678"

# Realistic multi-word page texts so str_fuzz scores high on identical content
PAGE_TEXTS = {
    "00000001": "The quick brown fox jumps over the lazy dog.",
    "00000002": "To be or not to be, that is the question.",
    "00000003": "It was the best of times, it was the worst of times.",
}


def make_zip(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create tmp_path/test.zip with the given filename->content mapping."""
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return zip_path


def make_pages_df(page_ids: list[str]) -> pl.DataFrame:
    """Build a minimal pages DataFrame from dot- or underscore-separated ids
    like 'work.00000001' or 'work_00000001', looked up against PAGE_TEXTS."""
    import re

    return pl.DataFrame(
        {
            "id": page_ids,
            "text": [
                PAGE_TEXTS[re.search(r"[0-9]+$", pid).group()] for pid in page_ids
            ],
        }
    )


@pytest.fixture
def aligned_zip(tmp_path):
    """Zip with all three PAGE_TEXTS entries using plain numeric filenames."""
    files = {f"{pid}.txt": text for pid, text in PAGE_TEXTS.items()}
    return make_zip(tmp_path, files)


@pytest.fixture
def pages_df():
    """Pages DataFrame with dot-separated ids for all three PAGE_TEXTS entries."""
    return make_pages_df(["work.00000001", "work.00000002", "work.00000003"])


# --- get_zip_textfiles ---


def test_get_zip_textfiles_returns_iterator(tmp_path):
    with ZipFile(make_zip(tmp_path, {"00000001.txt": "text"})) as zf:
        result = get_zip_textfiles(zf)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


def test_get_zip_textfiles_multiple(tmp_path):
    files = {"00000001.txt": "page one", "00000002.txt": "page two"}
    with ZipFile(make_zip(tmp_path, files)) as zf:
        assert dict(get_zip_textfiles(zf)) == {
            "00000001": "page one",
            "00000002": "page two",
        }


def test_get_zip_textfiles_skips_non_txt(tmp_path):
    files = {
        "00000001.txt": "page text",
        "image.jpg": "binary",
        "metadata.xml": "<xml/>",
    }
    with ZipFile(make_zip(tmp_path, files)) as zf:
        assert list(get_zip_textfiles(zf)) == [("00000001", "page text")]


def test_get_zip_textfiles_no_txt_files(tmp_path):
    # covers both empty zip and zip with only non-txt files
    with ZipFile(make_zip(tmp_path, {})) as zf:
        assert list(get_zip_textfiles(zf)) == []
    with ZipFile(make_zip(tmp_path, {"image.jpg": "binary"})) as zf:
        assert list(get_zip_textfiles(zf)) == []


def test_get_zip_textfiles_prefixed_filename(tmp_path):
    # OSU-style filenames: stem is preserved as-is
    with ZipFile(
        make_zip(tmp_path, {"OSU_32435051461309_00000602.txt": "page text"})
    ) as zf:
        assert list(get_zip_textfiles(zf)) == [
            ("OSU_32435051461309_00000602", "page text")
        ]


def test_get_zip_textfiles_utf8_content(tmp_path):
    content = "café naïve résumé"
    with ZipFile(make_zip(tmp_path, {"00000001.txt": content})) as zf:
        assert list(get_zip_textfiles(zf)) == [("00000001", content)]


# --- get_zipfile_pages ---


def test_get_zipfile_pages_sorted_by_order(tmp_path):
    # zip entry order (namelist) is not guaranteed to be page order; non-padded
    # names like 2/10/11/100 also sort differently by string than by number.
    # get_zipfile_pages must return rows sorted ascending by numeric order.
    files = {
        "100.txt": "page one hundred",
        "2.txt": "page two",
        "11.txt": "page eleven",
        "10.txt": "page ten",
    }
    with ZipFile(make_zip(tmp_path, files)) as zf:
        df = get_zipfile_pages(zf)

    assert df["order"].to_list() == [2, 10, 11, 100]


# --- add_zip_file_to_tar ---


def test_add_zip_file_to_tar(tmp_path):
    content = b"image data"
    zip_path = make_zip(tmp_path, {"00000001.jpg": content})
    tar_path = tmp_path / "images.tar"
    with ZipFile(zip_path) as zf:
        with tarfile.open(tar_path, "w") as tar:
            add_zip_file_to_tar(zf, "00000001.jpg", tar, "work_id/00000001.jpg")
    with tarfile.open(tar_path) as tar:
        assert tar.getnames() == ["work_id/00000001.jpg"]
        assert tar.extractfile("work_id/00000001.jpg").read() == content


# --- align_pages ---


def test_align_pages_good_match_returns_mapping(pages_df, aligned_zip):
    with ZipFile(aligned_zip) as zf:
        result = align_pages(WORK_ID, pages_df, zf)
    # mapping is keyed by the full page id (matches how process_ht_work looks it up)
    assert result == {
        "work.00000001": "00000001",
        "work.00000002": "00000002",
        "work.00000003": "00000003",
    }


def test_align_pages_low_match_falls_through_to_shifted(tmp_path):
    # Content differs entirely -> avg score is low -> falls through to
    # align_shifted_pages, which finds no matches and returns an empty
    # mapping, so align_pages returns None.
    # align_shifted_pages needs an `order` column and long-enough texts.
    pages_df = pl.DataFrame(
        {
            "id": ["work.00000001", "work.00000002", "work.00000003"],
            "order": [1, 2, 3],
            "text": [_long_text(f"alpha-{i}") for i in range(3)],
        }
    )
    zip_path = make_zip(
        tmp_path,
        {f"0000000{i + 1}.txt": _long_text(f"zzzzz-{i}-qqqqq") for i in range(3)},
    )
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) == {}


def test_align_pages_join_mismatch_returns_partial(tmp_path, pages_df):
    # Zip is missing one page -> join count mismatch is warned about but the
    # partial mapping for the pages that did join is still returned.
    zip_path = make_zip(
        tmp_path,
        {
            "00000001.txt": PAGE_TEXTS["00000001"],
            "00000002.txt": PAGE_TEXTS["00000002"],
        },
    )
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) == {
            "work.00000001": "00000001",
            "work.00000002": "00000002",
        }


def test_align_pages_insufficient_zip_pages(tmp_path, pages_df):
    # Zip has only one of the corpus's three pages -> partial mapping returned.
    zip_path = make_zip(tmp_path, {"00000001.txt": PAGE_TEXTS["00000001"]})
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) == {
            "work.00000001": "00000001",
        }


def test_align_pages_prefixed_filenames(tmp_path):
    # OSU-style zip filenames: page_id extracted from numeric suffix
    pages_df = make_pages_df(["work.00000001", "work.00000002"])
    zip_path = make_zip(
        tmp_path,
        {
            "OSU_32435051461309_00000001.txt": PAGE_TEXTS["00000001"],
            "OSU_32435051461309_00000002.txt": PAGE_TEXTS["00000002"],
        },
    )
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) == {
            "work.00000001": "OSU_32435051461309_00000001",
            "work.00000002": "OSU_32435051461309_00000002",
        }


# --- process_gale_work ---


def test_process_gale_work_missing_image_dir_yields_each_page(tmp_path):
    # when the volume image dir does not exist, every page must be yielded
    # individually (not the whole list as a single item)
    pages = [
        {"work_id": "CB0127060085", "id": "CB0127060085.0001", "text": "one"},
        {"work_id": "CB0127060085", "id": "CB0127060085.0002", "text": "two"},
    ]
    image_dir = tmp_path / "images"  # does not contain the volume dir
    image_dir.mkdir()
    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(process_gale_work("CB0127060085", pages, image_dir, tar))
    # each page is yielded as its own dict, unchanged, with no image_path
    assert result == pages
    assert all(isinstance(p, dict) for p in result)
    assert all("image_path" not in p for p in result)


def test_process_gale_work_adds_image_path_when_image_present(tmp_path):
    from corppa.utils.path_utils import get_gale_image_name, get_vol_dir

    vol_id = "CB0127060085"
    pages = [{"work_id": vol_id, "id": f"{vol_id}.0001", "text": "one"}]
    image_dir = tmp_path / "images"
    vol_img_dir = image_dir / get_vol_dir(vol_id)
    vol_img_dir.mkdir(parents=True)
    # create the expected Gale image file for page 1
    img_name = get_gale_image_name(vol_id, 1)
    (vol_img_dir / img_name).write_bytes(b"fake image data")

    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(process_gale_work(vol_id, pages, image_dir, tar))

    assert len(result) == 1
    assert result[0]["image_path"] == f"{vol_id}/{img_name}"


def test_process_gale_work_missing_image_file_omits_path(tmp_path):
    from corppa.utils.path_utils import get_vol_dir

    vol_id = "CB0127060085"
    pages = [{"work_id": vol_id, "id": f"{vol_id}.0001", "text": "one"}]
    image_dir = tmp_path / "images"
    # volume dir exists but the page image file is missing
    (image_dir / get_vol_dir(vol_id)).mkdir(parents=True)

    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(process_gale_work(vol_id, pages, image_dir, tar))

    assert len(result) == 1
    assert "image_path" not in result[0]


# --- open_ht_zipfile ---


def test_open_ht_zipfile_opens_existing(tmp_path):
    zip_path = tmp_path / "vol.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("00000001.tif", b"a")
    with open_ht_zipfile(zip_path) as ht_zip:
        assert ht_zip is not None
        assert ht_zip.namelist() == ["00000001.tif"]


def test_open_ht_zipfile_missing_path_yields_none(tmp_path):
    with open_ht_zipfile(tmp_path / "does-not-exist.zip") as ht_zip:
        assert ht_zip is None


def test_open_ht_zipfile_none_path_yields_none():
    with open_ht_zipfile(None) as ht_zip:
        assert ht_zip is None


# --- process_ht_work ---


def _make_ht_zip(tmp_path, htid_suffix, page_texts, with_images=True):
    """Build a HathiTrust-style zip at the path process_ht_work expects.
    page_texts maps zero-padded page filenames (e.g. '00000001') to text."""
    from corppa.utils.path_utils import encode_htid

    htid = f"test.{htid_suffix}"
    zip_dir = tmp_path / "HathiTrust" / encode_htid(htid)
    zip_dir.mkdir(parents=True)
    zip_path = zip_dir / f"{htid_suffix}.zip"
    with ZipFile(zip_path, "w") as zf:
        for name, text in page_texts.items():
            zf.writestr(f"{htid_suffix}/{name}.txt", text)
            if with_images:
                zf.writestr(f"{htid_suffix}/{name}.jpg", b"img-" + name.encode())
    return htid


def test_process_ht_work_no_zip_yields_pages_unchanged(tmp_path):
    htid_suffix = "12345678"
    work_id = f"test.{htid_suffix}"
    pages = [{"work_id": work_id, "id": f"{work_id}.00000001", "text": "hi"}]
    image_dir = tmp_path  # no HathiTrust zip present
    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(process_ht_work(work_id, pages, image_dir, tar))
    assert result == pages
    assert "image_path" not in result[0]


def test_process_ht_work_aligned_pages_get_image_paths(tmp_path):
    htid_suffix = "12345678"
    work_id = f"test.{htid_suffix}"
    _make_ht_zip(tmp_path, htid_suffix, PAGE_TEXTS, with_images=True)
    pages = [
        {"work_id": work_id, "id": f"{work_id}.{pid}", "text": text}
        for pid, text in PAGE_TEXTS.items()
    ]
    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(process_ht_work(work_id, pages, tmp_path, tar))
    # all pages returned, each with an image path in the tar
    assert len(result) == len(pages)
    assert all("image_path" in p for p in result)


def test_process_ht_work_does_not_drop_unaligned_pages(tmp_path):
    # a page with no alignment (page_basename is None) must still be yielded,
    # just without an image_path -- it should not silently disappear
    htid_suffix = "12345678"
    work_id = f"test.{htid_suffix}"
    _make_ht_zip(tmp_path, htid_suffix, PAGE_TEXTS, with_images=True)
    pages = [
        {"work_id": work_id, "id": f"{work_id}.{pid}", "text": text}
        for pid, text in PAGE_TEXTS.items()
    ]
    # add an extra corpus page that has no counterpart in the zip
    pages.append(
        {"work_id": work_id, "id": f"{work_id}.00000099", "text": "unmatched page"}
    )

    with patch(
        "corppa.utils.dataset_prep.align_pages",
        return_value={
            f"{work_id}.{pid}": pid for pid in PAGE_TEXTS
        },  # 00000099 intentionally absent
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(process_ht_work(work_id, pages, tmp_path, tar))

    # every input page is present in the output, including the unaligned one
    result_ids = [p["id"] for p in result]
    assert f"{work_id}.00000099" in result_ids
    assert len(result) == len(pages)
    # the unaligned page has no image_path
    unaligned = next(p for p in result if p["id"] == f"{work_id}.00000099")
    assert "image_path" not in unaligned


def test_process_ht_work_no_mapping_yields_pages_unchanged(tmp_path):
    # when align_pages returns no mapping, all pages are yielded without images
    htid_suffix = "12345678"
    work_id = f"test.{htid_suffix}"
    _make_ht_zip(tmp_path, htid_suffix, PAGE_TEXTS, with_images=True)
    pages = [
        {"work_id": work_id, "id": f"{work_id}.{pid}", "text": text}
        for pid, text in PAGE_TEXTS.items()
    ]
    with patch("corppa.utils.dataset_prep.align_pages", return_value={}):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(process_ht_work(work_id, pages, tmp_path, tar))
    assert [p["id"] for p in result] == [p["id"] for p in pages]
    assert all("image_path" not in p for p in result)


def test_process_ht_work_missing_image_warns_for_page_with_text(tmp_path, caplog):
    # page is aligned to a zip filename, but adding the image raises KeyError
    # (image absent from the zip); a page with text should warn and be yielded
    # without an image_path -- it must not be dropped
    htid_suffix = "12345678"
    work_id = f"test.{htid_suffix}"
    _make_ht_zip(tmp_path, htid_suffix, PAGE_TEXTS, with_images=True)
    pages = [
        {"work_id": work_id, "id": f"{work_id}.{pid}", "text": text}
        for pid, text in PAGE_TEXTS.items()
    ]
    with (
        patch(
            "corppa.utils.dataset_prep.align_pages",
            return_value={f"{work_id}.{pid}": pid for pid in PAGE_TEXTS},
        ),
        patch(
            "corppa.utils.dataset_prep.add_zip_file_to_tar",
            side_effect=KeyError("missing"),
        ),
        caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"),
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(process_ht_work(work_id, pages, tmp_path, tar))

    # every page is still yielded, none get an image_path
    assert [p["id"] for p in result] == [p["id"] for p in pages]
    assert all("image_path" not in p for p in result)
    # pages with text warn about the missing image
    assert "not found in zipfile but page has text; skipping" in caplog.text


def test_process_ht_work_missing_image_no_warn_for_empty_page(tmp_path, caplog):
    # when add_zip_file_to_tar raises KeyError for a page with no text,
    # the page is yielded without an image_path and without a warning
    htid_suffix = "12345678"
    work_id = f"test.{htid_suffix}"
    _make_ht_zip(tmp_path, htid_suffix, PAGE_TEXTS, with_images=True)
    # single page with only whitespace text
    pages = [{"work_id": work_id, "id": f"{work_id}.00000001", "text": "   "}]
    with (
        patch(
            "corppa.utils.dataset_prep.align_pages",
            return_value={f"{work_id}.00000001": "00000001"},
        ),
        patch(
            "corppa.utils.dataset_prep.add_zip_file_to_tar",
            side_effect=KeyError("missing"),
        ),
        caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"),
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(process_ht_work(work_id, pages, tmp_path, tar))

    assert [p["id"] for p in result] == [p["id"] for p in pages]
    assert "image_path" not in result[0]
    # no warning for a blank page missing its image
    assert "not found in zipfile but page has text" not in caplog.text


# --- process_work (dispatch) ---


def test_process_work_gale_dispatch(tmp_path):
    # a Gale work id (CB0.../CW0...) dispatches to process_gale_work
    work_id = "CB0127060085"
    pages = [{"work_id": work_id, "id": f"{work_id}.0001", "text": "p1"}]
    with (
        patch(
            "corppa.utils.dataset_prep.process_gale_work",
            return_value=iter(pages),
        ) as mock_gale,
        patch("corppa.utils.dataset_prep.process_ht_work") as mock_ht,
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(process_work(work_id, pages, tmp_path, tar))
    mock_gale.assert_called_once_with(work_id, pages, tmp_path, tar)
    mock_ht.assert_not_called()
    assert result == pages


def test_process_work_hathitrust_dispatch(tmp_path):
    # a HathiTrust work id (contains ".") dispatches to process_ht_work
    work_id = "test.12345678"
    pages = [{"work_id": work_id, "id": f"{work_id}.0001", "text": "p1"}]
    with (
        patch(
            "corppa.utils.dataset_prep.process_ht_work",
            return_value=iter(pages),
        ) as mock_ht,
        patch("corppa.utils.dataset_prep.process_gale_work") as mock_gale,
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(process_work(work_id, pages, tmp_path, tar))
    mock_ht.assert_called_once_with(work_id, pages, tmp_path, tar)
    mock_gale.assert_not_called()
    assert result == pages


def test_process_work_eebo_yields_pages_without_images(tmp_path):
    # an EEBO-TCP work id (begins with "A") has no images; pages pass through
    work_id = "A12345"
    pages = [{"work_id": work_id, "id": f"{work_id}.0001", "text": "p1"}]
    with (
        patch("corppa.utils.dataset_prep.process_gale_work") as mock_gale,
        patch("corppa.utils.dataset_prep.process_ht_work") as mock_ht,
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(process_work(work_id, pages, tmp_path, tar))
    mock_gale.assert_not_called()
    mock_ht.assert_not_called()
    # pages are yielded unchanged, with no image paths added
    assert result == pages
    assert all("image_path" not in p for p in result)


def test_process_work_unknown_source_warns_and_yields(tmp_path, caplog):
    # get_ppa_source raises for unrecognized ids; patch it to return an
    # unexpected source so we exercise the default branch
    work_id = "mystery-work"
    pages = [{"work_id": work_id, "id": f"{work_id}.0001", "text": "p1"}]
    with (
        patch(
            "corppa.utils.dataset_prep.get_ppa_source",
            return_value="SomethingElse",
        ),
        patch("corppa.utils.dataset_prep.process_gale_work") as mock_gale,
        patch("corppa.utils.dataset_prep.process_ht_work") as mock_ht,
        caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"),
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(process_work(work_id, pages, tmp_path, tar))
    mock_gale.assert_not_called()
    mock_ht.assert_not_called()
    # pages are not dropped, and the unknown source is surfaced as a warning
    assert result == pages
    assert "unknown source 'SomethingElse'" in caplog.text


def test_process_work_ht1930_dispatch(tmp_path):
    # a HathiTrust 1930 work dispatches to the 1930-specific path instead of
    # the (text-aligning) process_ht_work, passing along its digital page range
    work_id = "test.12345678"
    pages = [{"work_id": work_id, "id": f"{work_id}.0001", "order": 1, "text": "p1"}]
    with (
        patch(
            "corppa.utils.dataset_prep.process_ht1930_work",
            return_value=iter(pages),
        ) as mock_ht1930,
        patch("corppa.utils.dataset_prep.process_ht_work") as mock_ht,
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(
                process_work(
                    work_id,
                    pages,
                    tmp_path,
                    tar,
                    ht1930_work_ids={work_id: "1-10"},
                )
            )
    mock_ht1930.assert_called_once_with(
        work_id, pages, tmp_path, tar, digital_pages="1-10"
    )
    mock_ht.assert_not_called()
    assert result == pages


def test_process_work_hathitrust_not_1930_uses_ht_work(tmp_path):
    # a HathiTrust work NOT in the 1930 set uses the standard path even when
    # other works are flagged as 1930
    work_id = "test.12345678"
    pages = [{"work_id": work_id, "id": f"{work_id}.0001", "order": 1, "text": "p1"}]
    with (
        patch(
            "corppa.utils.dataset_prep.process_ht_work",
            return_value=iter(pages),
        ) as mock_ht,
        patch("corppa.utils.dataset_prep.process_ht1930_work") as mock_ht1930,
    ):
        with tarfile.open(tmp_path / "out.tar", "w") as tar:
            result = list(
                process_work(
                    work_id,
                    pages,
                    tmp_path,
                    tar,
                    ht1930_work_ids={"other.99": None},
                )
            )
    mock_ht.assert_called_once_with(work_id, pages, tmp_path, tar)
    mock_ht1930.assert_not_called()
    assert result == pages


# --- zip_image_filenames ---


def test_zip_image_filenames_maps_numeric_stem(tmp_path):
    zip_path = tmp_path / "vol.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("00000001.tif", b"a")
        zf.writestr("00000002.jpg", b"b")
        # non-image files are ignored
        zf.writestr("00000003.txt", b"text")
        zf.writestr("notes.xml", b"meta")
    with ZipFile(zip_path) as zf:
        result = zip_image_filenames(zf)
    assert result == {1: "00000001.tif", 2: "00000002.jpg"}


def test_zip_image_filenames_skips_non_numeric_stem(tmp_path):
    # a stray image with a non-numeric filename must be skipped, not crash
    zip_path = tmp_path / "vol.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("00000001.tif", b"a")
        zf.writestr("cover.jpg", b"cover")
    with ZipFile(zip_path) as zf:
        result = zip_image_filenames(zf)
    assert result == {1: "00000001.tif"}


# --- process_ht1930_work ---


def _make_ht1930_zip(tmp_path, zip_name, page_nums, ext=".tif"):
    """Build an image-only HathiTrust-1930 zip (flat, numeric image filenames,
    no OCR text) under the image_dir/HathiTrust-1930/ directory."""
    ht1930_dir = tmp_path / "HathiTrust-1930"
    ht1930_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ht1930_dir / zip_name
    with ZipFile(zip_path, "w") as zf:
        for num in page_nums:
            name = f"{num:08d}"
            zf.writestr(f"{name}{ext}", b"img-" + name.encode())
    return zip_path


def _ht1930_pages(work_id, orders):
    """Build page dicts with id + order for a 1930 full work."""
    return [
        {
            "work_id": work_id,
            "id": f"{work_id}.{n:08d}",
            "order": n,
            "text": "",
        }
        for n in orders
    ]


def test_process_ht1930_no_zip_yields_pages_unchanged(tmp_path, caplog):
    work_id = "test.12345678"
    pages = _ht1930_pages(work_id, [1])
    # HathiTrust-1930 dir exists but has no matching zip
    (tmp_path / "HathiTrust-1930").mkdir()
    with (
        caplog.at_level("ERROR", logger="corppa.utils.dataset_prep"),
        tarfile.open(tmp_path / "out.tar", "w") as tar,
    ):
        result = list(process_ht1930_work(work_id, pages, tmp_path, tar))
    assert result == pages
    assert "image_path" not in result[0]
    assert "Expected exactly one zipfile" in caplog.text


def test_process_ht1930_maps_images_by_order(tmp_path):
    work_id = "test.12345678"
    # full-work zip names are "{htid-dashes}-{HT id}.zip" (trailing number is an
    # unrelated HT-assigned id), matched by a wildcard on the htid prefix
    _make_ht1930_zip(tmp_path, "test-12345678-1788450816.zip", [1, 2, 3])
    pages = _ht1930_pages(work_id, [1, 2, 3])
    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(process_ht1930_work(work_id, pages, tmp_path, tar))
        tar_names = tar.getnames()
    # every page gets an image path pointing into the tar
    assert len(result) == len(pages)
    assert all("image_path" in p for p in result)
    from corppa.utils.path_utils import encode_htid

    encoded = encode_htid(work_id)
    assert result[0]["image_path"] == f"{encoded}/{work_id}.00000001.tif"
    assert f"{encoded}/{work_id}.00000001.tif" in tar_names


def test_process_ht1930_excerpt_uses_first_digital_page_in_name(tmp_path):
    # excerpt zip names are "{htid}-{first_page}-{last_page}-{HT id}.zip"; the
    # first digital page is matched, the trailing segments are wildcarded
    work_id = "test.12345678-p5"
    _make_ht1930_zip(tmp_path, "test-12345678-5-6-1788473798.zip", [5, 6])
    pages = _ht1930_pages("test.12345678", [5, 6])
    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(
            process_ht1930_work(work_id, pages, tmp_path, tar, digital_pages="5-6")
        )
    assert all("image_path" in p for p in result)


def test_process_ht1930_does_not_drop_unmatched_pages(tmp_path):
    work_id = "test.12345678"
    # zip has images for orders 1 and 2 only
    _make_ht1930_zip(tmp_path, "test-12345678-1788450816.zip", [1, 2])
    pages = _ht1930_pages(work_id, [1, 2, 99])
    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(process_ht1930_work(work_id, pages, tmp_path, tar))
    # page with order 99 (no matching image) is still yielded, without an image_path
    result_ids = [p["id"] for p in result]
    assert f"{work_id}.00000099" in result_ids
    unmatched = next(p for p in result if p["id"] == f"{work_id}.00000099")
    assert "image_path" not in unmatched


def test_process_ht1930_no_images_yields_pages_unchanged(tmp_path, caplog):
    work_id = "test.12345678"
    # build a zip with no image files (empty page list)
    _make_ht1930_zip(tmp_path, "test-12345678-1788450816.zip", [])
    pages = _ht1930_pages(work_id, [1])
    with (
        caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"),
        tarfile.open(tmp_path / "out.tar", "w") as tar,
    ):
        result = list(process_ht1930_work(work_id, pages, tmp_path, tar))
    assert result == pages
    assert all("image_path" not in p for p in result)
    assert "no images found in image-only zipfile" in caplog.text


def test_process_ht1930_warns_when_no_page_order_matches(tmp_path, caplog):
    # zip has images, but their numbering does not line up with any page order
    # (e.g. relative vs absolute digital sequence); warn instead of silently
    # dropping every image
    work_id = "test.12345678"
    _make_ht1930_zip(tmp_path, "test-12345678-1788450816.zip", [1, 2, 3])
    # pages are numbered 101-103, which do not exist in the zip
    pages = _ht1930_pages(work_id, [101, 102, 103])
    with (
        caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"),
        tarfile.open(tmp_path / "out.tar", "w") as tar,
    ):
        result = list(process_ht1930_work(work_id, pages, tmp_path, tar))
    assert result == pages
    assert all("image_path" not in p for p in result)
    assert "none matched a page order" in caplog.text


def test_process_ht1930_full_work_prefix_not_confused_with_other_volume(tmp_path):
    # the htid-prefix wildcard for a full work must not match a different
    # volume whose htid happens to start with the same characters
    work_id = "test.12345678"
    _make_ht1930_zip(tmp_path, "test-12345678-1788450816.zip", [1, 2])
    # a different, longer htid that shares a leading substring
    _make_ht1930_zip(tmp_path, "test-123456789999-42.zip", [1, 2])
    pages = _ht1930_pages(work_id, [1, 2])
    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(process_ht1930_work(work_id, pages, tmp_path, tar))
    # exactly the matching volume zip is selected; images are added
    assert all("image_path" in p for p in result)


def test_process_ht1930_ambiguous_zip_match_yields_pages(tmp_path, caplog):
    # if the htid-prefix wildcard matches more than one zip, we can't safely
    # choose; pages are yielded without images and an error is logged
    work_id = "test.12345678"
    _make_ht1930_zip(tmp_path, "test-12345678-1788450816.zip", [1, 2])
    _make_ht1930_zip(tmp_path, "test-12345678-9999999999.zip", [1, 2])
    pages = _ht1930_pages(work_id, [1, 2])
    with (
        caplog.at_level("ERROR", logger="corppa.utils.dataset_prep"),
        tarfile.open(tmp_path / "out.tar", "w") as tar,
    ):
        result = list(process_ht1930_work(work_id, pages, tmp_path, tar))
    assert all("image_path" not in p for p in result)
    assert "Expected exactly one zipfile" in caplog.text


def test_process_ht1930_excerpt_first_page_override(tmp_path):
    # the known excerpt override supplies a first-page number that cannot be
    # derived from the digital page range (missing pages in the scan)
    work_id = "mdp.39015030593423-p165"
    _make_ht1930_zip(tmp_path, "mdp-39015030593423-193-194-1788473798.zip", [193, 194])
    pages = _ht1930_pages("mdp.39015030593423", [193, 194])
    with tarfile.open(tmp_path / "out.tar", "w") as tar:
        result = list(
            # digital_pages here would derive 165, but the override forces 193
            process_ht1930_work(work_id, pages, tmp_path, tar, digital_pages="165-166")
        )
    assert all("image_path" in p for p in result)


# --- find_corpus_file / get_ht1930_work_ids ---


def test_find_corpus_file_prefers_first_existing(tmp_path):
    (tmp_path / "ppa_pages.jsonl.gz").write_text("")
    # only the compressed file exists; it should be returned
    assert find_corpus_file(tmp_path, ["ppa_pages.jsonl", "ppa_pages.jsonl.gz"]) == (
        tmp_path / "ppa_pages.jsonl.gz"
    )
    # uncompressed preferred when both exist
    (tmp_path / "ppa_pages.jsonl").write_text("")
    assert find_corpus_file(tmp_path, ["ppa_pages.jsonl", "ppa_pages.jsonl.gz"]) == (
        tmp_path / "ppa_pages.jsonl"
    )


def test_find_corpus_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="None of the expected files"):
        find_corpus_file(tmp_path, ["ppa_pages.jsonl"])


def test_get_ht1930_work_ids_csv(tmp_path):
    meta = tmp_path / "ppa_metadata.csv"
    meta.write_text(
        "work_id,pub_year,source,pages_digital\n"
        "ht.old,1850,HathiTrust,\n"
        "ht.new,1930,HathiTrust,\n"
        "ht.excerpt,1930,HathiTrust,5-10\n"
        "gale.new,1930,Gale,\n"
        "ht.1931,1931,HathiTrust,\n"
    )
    result = get_ht1930_work_ids(meta)
    # only HathiTrust works published in 1930; value is the digital page range
    assert result == {"ht.new": None, "ht.excerpt": "5-10"}


def test_get_ht1930_work_ids_json(tmp_path):
    meta = tmp_path / "ppa_metadata.json"
    meta.write_text(
        '[{"work_id": "ht.old", "pub_year": 1899, "source": "HathiTrust", "pages_digital": null},'
        ' {"work_id": "ht.new", "pub_year": 1930, "source": "HathiTrust", "pages_digital": null},'
        ' {"work_id": "gale.new", "pub_year": 1930, "source": "Gale", "pages_digital": null}]'
    )
    result = get_ht1930_work_ids(meta)
    assert result == {"ht.new": None}


def test_get_ht1930_work_ids_unsupported_format(tmp_path):
    meta = tmp_path / "ppa_metadata.txt"
    meta.write_text("work_id,pub_year,source,pages_digital\nht.new,1930,HathiTrust,\n")
    with pytest.raises(ValueError, match="Unsupported metadata format"):
        get_ht1930_work_ids(meta)


# --- longest_increasing_subseq ---


def test_longest_increasing_subseq_empty():
    result = longest_increasing_subseq(np.array([], dtype=int))
    assert result.tolist() == []


def test_longest_increasing_subseq_already_increasing():
    # every element is part of the (only) increasing run
    values = np.array([10, 11, 12, 13])
    assert longest_increasing_subseq(values).tolist() == [0, 1, 2, 3]


def test_longest_increasing_subseq_drops_backwards_jump():
    # values: 10, 11, 5, 12 -> the '5' at index 2 breaks the sequence and is
    # dropped in favor of the longer run 10,11,12
    values = np.array([10, 11, 5, 12])
    assert longest_increasing_subseq(values).tolist() == [0, 1, 3]


def test_longest_increasing_subseq_strict_drops_duplicates():
    # equal values must not both be kept (a zip page claimed twice); strictness
    # keeps only one of the repeated 11s
    values = np.array([10, 11, 11, 12])
    result = longest_increasing_subseq(values)
    kept = values[result]
    # strictly increasing and no repeats
    assert kept.tolist() == [10, 11, 12]
    assert len(set(kept.tolist())) == len(kept)


def test_longest_increasing_subseq_keeps_longest_run():
    # a single large early value should not suppress a longer later run
    values = np.array([100, 1, 2, 3, 4])
    result = longest_increasing_subseq(values)
    assert values[result].tolist() == [1, 2, 3, 4]


# --- align_shifted_pages ---


def _long_text(seed: str, length: int = 700) -> str:
    """Repeat `seed` until the result exceeds `length` chars so it survives
    the >600-char filter inside align_shifted_pages, while staying distinct
    from other seeds (each repeated seed produces a unique long string)."""
    reps = (length // len(seed)) + 2
    return (seed + " ") * reps


def _make_shifted_frames(page_orders, zip_orders, seeds):
    """Build (pages_df, zip_pages_df) where pages_df.order[i] and
    zip_pages_df.order[i] share the same long text derived from seeds[i]."""
    texts = [_long_text(s) for s in seeds]
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{o:08d}" for o in page_orders],
            "order": page_orders,
            "text": texts,
        }
    )
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": [f"{o:08d}" for o in zip_orders],
            "order": zip_orders,
            "text": texts,
        }
    )
    return pages_df, zip_pages_df


def test_align_shifted_pages_consistent_shift():
    # pages orders 1..5 correspond to zip orders 11..15 (uniform shift = +10);
    # every page should map to its shifted zip counterpart, including the
    # first and last anchors.
    seeds = [f"chapter-{i}-unique-content" for i in range(5)]
    pages_df, zip_pages_df = _make_shifted_frames(
        page_orders=list(range(1, 6)),
        zip_orders=list(range(11, 16)),
        seeds=seeds,
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    assert mapping == {
        "work.00000001": "00000011",
        "work.00000002": "00000012",
        "work.00000003": "00000013",
        "work.00000004": "00000014",
        "work.00000005": "00000015",
    }


def test_align_shifted_pages_duplicate_zip_not_double_claimed():
    # Two original pages (orders 2 and 4) share identical boilerplate text, so
    # both would independently pick the same zip page as their best match.
    # The strict-increasing anchor filter must prevent that zip page from being
    # claimed twice; the resulting alignment must not map two pages to the same
    # zip filename.
    boiler = _long_text("this-page-is-repeated-boilerplate")
    unique_seeds = [
        "chapter-one-unique",
        "chapter-two-unique",
        "chapter-three-unique",
    ]
    # pages 1,3,5 have unique text; pages 2,4 share the same boilerplate
    page_texts = [
        _long_text(unique_seeds[0]),
        boiler,
        _long_text(unique_seeds[1]),
        boiler,
        _long_text(unique_seeds[2]),
    ]
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{o:08d}" for o in range(1, 6)],
            "order": list(range(1, 6)),
            "text": page_texts,
        }
    )
    # zip has the same content at a uniform +10 shift
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": [f"{o:08d}" for o in range(11, 16)],
            "order": list(range(11, 16)),
            "text": page_texts,
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    filenames = result["page_filename"].drop_nulls().to_list()
    # no zip filename is assigned to more than one original page
    assert len(filenames) == len(set(filenames))


def test_align_shifted_pages_fill_boundary_duplicate_resolved():
    # A shift-boundary duplicate that LIS cannot catch: anchors are consistent,
    # but the forward/back-fill makes two adjacent pages resolve to the same
    # aligned zip page. The score-based dedup must keep the real long-text match
    # and null the (short, filled) loser so no zip page is assigned twice.
    seeds = [f"unique-content-{i}" for i in range(4)]
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{o:08d}" for o in range(1, 5)],
            "order": [1, 2, 3, 4],
            # pages 2,3 are short -> shift is filled from neighboring anchors
            "text": ["short a", "short b", _long_text(seeds[2]), _long_text(seeds[3])],
        }
    )
    # anchor at order 1 -> zip 11 (shift +10); anchor at order 4 -> zip 13 (shift +9)
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": ["00000011", "00000012", "00000013"],
            "order": [11, 12, 13],
            "text": [_long_text(seeds[2]), "zip junk", _long_text(seeds[3])],
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    filenames = [f for f in mapping.values() if f is not None]
    # strict 1:1: no zip filename assigned to more than one page
    assert len(filenames) == len(set(filenames))
    # the genuine long-text match keeps the contested filename
    assert mapping["work.00000004"] == "00000013"


def test_align_shifted_pages_skips_scoring_when_no_duplicates():
    # dedup only needs to score pages that share a zip filename; when the mapping
    # is already 1:1, str_fuzz should not be called at all (non-detailed path).
    seeds = [f"chapter-{i}-unique-content" for i in range(4)]
    pages_df, zip_pages_df = _make_shifted_frames(
        page_orders=[1, 2, 3, 4], zip_orders=[11, 12, 13, 14], seeds=seeds
    )

    with patch("corppa.utils.dataset_prep.pds.str_fuzz") as mock_fuzz:
        align_shifted_pages(pages_df, zip_pages_df)

    mock_fuzz.assert_not_called()


def test_align_shifted_pages_includes_head_pages():
    # Pages before the first anchor are typically short pages that got
    # filtered out. They must still be included in the mapping via the
    # first anchor's shift.
    seeds = [f"chapter-{i}-unique-content" for i in range(6)]
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": [f"{11 + i:08d}" for i in range(6)],
            "order": [11 + i for i in range(6)],
            "text": [_long_text(s) for s in seeds],
        }
    )
    # page 1 is short (below the 600-char filter); pages 2-6 are long and
    # share text with zip pages at orders 12-16 (uniform shift = +10)
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{1 + i:08d}" for i in range(6)],
            "order": [1 + i for i in range(6)],
            "text": ["short leading page"] + [_long_text(s) for s in seeds[1:]],
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    # page 1 (short, before the first anchor) is mapped via the anchor's shift
    assert mapping["work.00000001"] == "00000011"
    # last page still covered by the tail fix
    assert mapping["work.00000006"] == "00000016"


def test_align_shifted_pages_returns_id_and_filename_columns():
    seeds = [f"page-{i}-content" for i in range(3)]
    pages_df, zip_pages_df = _make_shifted_frames([1, 2, 3], [5, 6, 7], seeds)

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    assert set(result.columns) >= {"id", "page_filename"}


def test_align_shifted_pages_no_content_match():
    # No shared content between pages and zip -> no page clears the cutoff
    pages_df, _ = _make_shifted_frames(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [f"alpha-{i}" for i in range(5)],
    )
    _, zip_pages_df = _make_shifted_frames(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [f"zzzzz-{i}-qqqqq" for i in range(5)],
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result.is_empty()


def test_align_shifted_pages_small_df_uses_all_anchors():
    # Only 2 long pages in pages_df -> can't sample first/middle/last without
    # duplicates; all long pages get used as anchors, and if they agree on a
    # shift, every original page (short or long) gets mapped.
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": [f"{11 + i:08d}" for i in range(3)],
            "order": [11, 12, 13],
            "text": [_long_text(f"chapter-{i}") for i in range(3)],
        }
    )
    # pages 1 and 2 are long (used as anchors), page 3 is short
    pages_df = pl.DataFrame(
        {
            "id": ["work.00000001", "work.00000002", "work.00000003"],
            "order": [1, 2, 3],
            "text": [
                _long_text("chapter-0"),
                _long_text("chapter-1"),
                "short trailing page",
            ],
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    # all three pages mapped via the shared shift (+10)
    assert mapping == {
        "work.00000001": "00000011",
        "work.00000002": "00000012",
        "work.00000003": "00000013",
    }


def test_align_shifted_pages_single_long_anchor():
    # Only one long page; the single-anchor shift is applied to every
    # original page via the head-chunk-extended-to-end branch.
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": ["00000011", "00000012", "00000013"],
            "order": [11, 12, 13],
            "text": [_long_text(f"chapter-{i}") for i in range(3)],
        }
    )
    pages_df = pl.DataFrame(
        {
            "id": ["work.00000001", "work.00000002", "work.00000003"],
            "order": [1, 2, 3],
            "text": ["short", _long_text("chapter-1"), "also short"],
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    assert mapping == {
        "work.00000001": "00000011",
        "work.00000002": "00000012",
        "work.00000003": "00000013",
    }


def test_align_shifted_pages_all_pages_short_returns_empty():
    # Every page below the 600-char filter -> no long-enough anchors.
    # Short-chunk guard returns an empty mapping instead of crashing.
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{i:08d}" for i in range(1, 4)],
            "order": [1, 2, 3],
            "text": ["short one", "short two", "short three"],
        }
    )
    _, zip_pages_df = _make_shifted_frames(
        [1, 2, 3], [1, 2, 3], ["short one", "short two", "short three"]
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    assert result.is_empty()


def test_align_shifted_pages_monotonic_gap_no_warning(caplog):
    # two segments with different (increasing) shifts leave a gap in the aligned
    # orders (11,12 then 20,21). gaps are fine as long as order is preserved,
    # so this should NOT warn about non-monotonic order.
    seeds = [f"chapter-{i}-unique-content" for i in range(4)]
    pages_df, zip_pages_df = _make_shifted_frames(
        page_orders=[1, 2, 3, 4],
        zip_orders=[11, 12, 20, 21],
        seeds=seeds,
    )

    with caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"):
        align_shifted_pages(pages_df, zip_pages_df)

    assert "not monotonic" not in caplog.text


def test_align_shifted_pages_logs_unmatched_pages(caplog):
    # pages 1-4 shift +10 -> aligned orders 11,12,13,14, but the zip only has
    # 11,12,13; page 4 aligns to a missing zip page and gets no filename.
    # the unmatched count is reported as part of the shift summary log line.
    seeds = [f"chapter-{i}-unique-content" for i in range(4)]
    pages_df, zip_pages_df = _make_shifted_frames(
        page_orders=[1, 2, 3, 4],
        zip_orders=[11, 12, 13, 14],
        seeds=seeds,
    )
    # drop the last zip page so page 4 has nothing to align to
    zip_pages_df = zip_pages_df.head(3)

    with caplog.at_level("INFO", logger="corppa.utils.dataset_prep"):
        align_shifted_pages(pages_df, zip_pages_df)

    assert "1 page unmatched" in caplog.text


def test_align_shifted_pages_non_monotonic_warns(caplog):
    # LIS filters anchors to a strictly increasing (by zip order) run, so a purely
    # backwards tail (e.g. 11,12 then 3,4) is now dropped rather than aligned.
    # A non-monotonic *matched* result can still slip through fill/dedup when the
    # tail zip orders decrease within an otherwise-increasing anchor set; here
    # 10,11,12 then 15,14,13 leaves later pages aligning to earlier zip pages,
    # which the monotonic sanity-check should warn about.
    seeds = [f"chapter-{i}-unique-content" for i in range(6)]
    pages_df, zip_pages_df = _make_shifted_frames(
        page_orders=[1, 2, 3, 4, 5, 6],
        zip_orders=[10, 11, 12, 15, 14, 13],
        seeds=seeds,
    )

    with caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"):
        align_shifted_pages(pages_df, zip_pages_df)

    assert "aligned page order is not monotonic" in caplog.text


def test_align_shifted_pages_independent_of_zip_row_order():
    # Regression: LIS anchoring must depend on zip page *order* values, not on the
    # row position of the zip pages in the dataframe. Non-zero-padded page orders
    # (2, 10, 11, 100) sort differently by filename string than by number, so
    # feeding the zip rows in filename (lexicographic) order previously corrupted
    # the anchor selection. The mapping must be identical regardless of row order.
    seeds = [f"chapter-{i}-unique-content" for i in range(4)]
    page_orders = [1, 2, 3, 4]
    zip_orders = [2, 10, 11, 100]
    pages_df, zip_asc = _make_shifted_frames(page_orders, zip_orders, seeds)
    # scramble the zip rows into filename (lexicographic) order: 10, 100, 11, 2
    zip_lex = zip_asc.sort("page_filename")

    result_asc = align_shifted_pages(pages_df, zip_asc)
    result_lex = align_shifted_pages(pages_df, zip_lex)

    mapping_asc = dict(result_asc.select(["id", "page_filename"]).iter_rows())
    mapping_lex = dict(result_lex.select(["id", "page_filename"]).iter_rows())
    # both row orders produce the same, fully-aligned mapping
    assert mapping_asc == mapping_lex
    assert mapping_asc == {
        "work.00000001": "00000002",
        "work.00000002": "00000010",
        "work.00000003": "00000011",
        "work.00000004": "00000100",
    }


def test_align_pages_underscore_page_id(aligned_zip):
    # Corpus page ids use underscore separator instead of dot
    pages_df = make_pages_df(["work_00000001", "work_00000002", "work_00000003"])
    with ZipFile(aligned_zip) as zf:
        result = align_pages(WORK_ID, pages_df, zf)
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "work_00000001",
        "work_00000002",
        "work_00000003",
    }


# --- detailed alignment review + visualization ---


def test_align_shifted_pages_detailed_schema():
    # detailed=True returns the raw alignment frame (no derived review fields)
    seeds = [f"chapter-{i}-unique-content" for i in range(5)]
    pages_df, zip_pages_df = _make_shifted_frames(
        page_orders=list(range(1, 6)),
        zip_orders=list(range(11, 16)),
        seeds=seeds,
    )

    result = align_shifted_pages(pages_df, zip_pages_df, detailed=True)

    # expected core columns are present (raw text is carried through;
    # derived review fields are added by review_alignment, not here)
    assert {
        "id",
        "order",
        "aligned_order",
        "page_filename",
        "cdist_best_score",
        "shift",
        "inferred_shift",
        "text",
        "zip_text",
    } <= set(result.columns)
    # derived review-only fields (incl. is_anchor/is_matched flags) are NOT added
    # by align_shifted_pages
    for col in ("is_anchor", "is_matched", "text_len", "text_snippet", "match_score"):
        assert col not in result.columns
    # all five pages align (uniform +10 shift): every page has a filename + shift
    assert result.height == 5
    assert result["page_filename"].null_count() == 0
    assert result["shift"].null_count() == 0
    # mapped order is original order + 10
    assert (result["aligned_order"] - result["order"]).unique().to_list() == [10]
    # every scored (long) page carries a cdist best score (identical text => 100)
    assert result["cdist_best_score"].null_count() == 0
    assert result["cdist_best_score"].min() == 100.0


def test_align_shifted_pages_detailed_marks_inferred_page():
    # a short leading page is filled (not an anchor) but still matched
    seeds = [f"chapter-{i}-unique-content" for i in range(5)]
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{o:08d}" for o in range(1, 6)],
            "order": list(range(1, 6)),
            "text": ["short lead"] + [_long_text(s) for s in seeds[1:]],
        }
    )
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": [f"{o:08d}" for o in range(11, 16)],
            "order": list(range(11, 16)),
            "text": ["zip junk"] + [_long_text(s) for s in seeds[1:]],
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df, detailed=True).sort("order")

    row1 = result.row(0, named=True)
    # first page is short: filled via neighbor shift, so it has no trusted shift
    # (not an anchor) but still resolves to a zip page (matched)
    assert row1["shift"] is None
    assert row1["page_filename"] is not None


def test_review_alignment_text_snippet(tmp_path):
    # review_alignment derives the snippet: first non-empty line, whitespace-
    # collapsed and truncated
    long_first_line = "This is the opening line of the page " * 5  # > 80 chars
    text = f"   \n\n{long_first_line}\nsecond line\n" + ("filler word " * 60)
    pages = [{"id": "work.00000001", "order": 1, "text": text}]
    zip_path = make_zip(tmp_path, {"00000011.txt": text})

    result = review_alignment("work", pages, zip_path)

    snippet = result["text_snippet"][0]
    # leading blank lines stripped; only the first line is used (no "second line")
    assert snippet.startswith("This is the opening line")
    assert "second line" not in snippet
    # truncated to the configured length plus a single-char ellipsis
    assert len(snippet) == dataset_prep.TEXT_SNIPPET_LEN + 1
    assert snippet.endswith("…")
    # matched zip snippet is populated too (same text here)
    assert result["zip_text_snippet"][0] == snippet


def test_align_shifted_pages_detailed_empty_has_diagnostic_schema():
    # no content match -> empty frame, but with the detailed schema so callers
    # can rely on the columns existing
    pages_df, zip_pages_df = _make_shifted_frames([1, 2], [1, 2], ["aaa", "bbb"])
    # overwrite with non-matching zip text
    zip_pages_df = zip_pages_df.with_columns(text=pl.lit("no shared content here"))

    result = align_shifted_pages(pages_df, zip_pages_df, detailed=True)

    assert result.is_empty()
    assert {"page_filename", "shift", "aligned_order"} <= set(result.columns)


def test_review_alignment_from_pages_and_zip(tmp_path):
    # review_alignment accepts page dicts + a zip path and returns the detailed frame
    seeds = [f"chapter-{i}-unique-content" for i in range(4)]
    pages = [
        {"id": f"work.{o:08d}", "order": o, "text": _long_text(seeds[o - 1])}
        for o in range(1, 5)
    ]
    zip_files = {f"{10 + o:08d}.txt": _long_text(seeds[o - 1]) for o in range(1, 5)}
    zip_path = make_zip(tmp_path, zip_files)

    result = review_alignment("work", pages, zip_path)

    assert result["is_matched"].all()
    # uniform +10 shift recovered
    assert (result["aligned_order"] - result["order"]).unique().to_list() == [10]


def test_review_alignment_missing_zip_raises(tmp_path):
    pages = [{"id": "work.00000001", "order": 1, "text": "hi"}]
    with pytest.raises(FileNotFoundError, match="zip file not found"):
        review_alignment("work", pages, tmp_path / "missing.zip")


def test_review_alignment_adds_derived_fields(tmp_path):
    # review_alignment adds lengths, snippets, and an aligned-page match_score
    # (comparable to cdist_best_score) on top of the raw alignment frame
    seeds = [f"chapter-{i}-unique-content" for i in range(3)]
    pages = [
        {"id": f"work.{o:08d}", "order": o, "text": _long_text(seeds[o - 1])}
        for o in range(1, 4)
    ]
    zip_files = {f"{10 + o:08d}.txt": _long_text(seeds[o - 1]) for o in range(1, 4)}
    zip_path = make_zip(tmp_path, zip_files)

    result = review_alignment("work", pages, zip_path)

    assert {
        "text_len",
        "zip_text_len",
        "text_snippet",
        "zip_text_snippet",
        "match_score",
    } <= set(result.columns)
    # identical text on both sides -> aligned match_score is 100 (0-100 scale),
    # matching cdist_best_score
    assert result["match_score"].to_list() == [100.0, 100.0, 100.0]
    assert result["cdist_best_score"].to_list() == [100.0, 100.0, 100.0]


def test_review_alignment_derives_order_when_missing(tmp_path):
    # when pages lack an explicit order column, it is derived from the id suffix
    seeds = [f"chapter-{i}-unique-content" for i in range(3)]
    pages = [
        {"id": f"work.{o:08d}", "text": _long_text(seeds[o - 1])} for o in range(1, 4)
    ]
    zip_files = {f"{10 + o:08d}.txt": _long_text(seeds[o - 1]) for o in range(1, 4)}
    zip_path = make_zip(tmp_path, zip_files)

    result = review_alignment("work", pages, zip_path)

    assert result["order"].to_list() == [1, 2, 3]


def test_get_ht_zipfile_path():
    # path is image_dir/HathiTrust/<encoded>/<suffix>.zip
    path = get_ht_zipfile_path("htid:test.12345678", Path("/img"))
    assert path.parts[-3] == "HathiTrust"
    assert path.suffix == ".zip"
    # suffix filename matches the last dotted segment of the encoded id
    assert path.stem == path.parts[-2].split(".")[-1]


def test_plot_alignment_builds_chart(tmp_path):
    seeds = [f"chapter-{i}-unique-content" for i in range(4)]
    pages = [
        {"id": f"work.{o:08d}", "order": o, "text": _long_text(seeds[o - 1])}
        for o in range(1, 5)
    ]
    zip_files = {f"{10 + o:08d}.txt": _long_text(seeds[o - 1]) for o in range(1, 5)}
    review_df = review_alignment("work", pages, make_zip(tmp_path, zip_files))

    chart = plot_alignment(review_df)

    # a valid vega-lite spec with the line + points layers
    spec = chart.to_dict()
    assert "layer" in spec
    assert len(spec["layer"]) == 2
    # long-form data: an "original" row point per page plus a "mapped zip" row
    # point per matched page (all 4 match here) -> 8 rows
    rows = chart.data["row"].to_list()
    assert rows.count("original") == 4
    assert rows.count("mapped zip") == 4
    # hover info + snippet carried into the plot data
    assert {"info", "snippet"} <= set(chart.data.columns)
    # info line summarizes the order mapping
    assert chart.data["info"][0].startswith("page ")


def test_plot_alignment_unmatched_pages_have_no_zip_point(tmp_path):
    # an unmatched page appears only on the original row (no zip point/line)
    seeds = [f"chapter-{i}-unique-content" for i in range(4)]
    pages = [
        {"id": f"work.{o:08d}", "order": o, "text": _long_text(seeds[o - 1])}
        for o in range(1, 5)
    ]
    # zip only has 3 pages, so one original page can't map -> unmatched
    zip_files = {f"{10 + o:08d}.txt": _long_text(seeds[o - 1]) for o in range(1, 4)}
    review_df = review_alignment("work", pages, make_zip(tmp_path, zip_files))
    n_matched = int(review_df["is_matched"].sum())

    chart = plot_alignment(review_df)
    rows = chart.data["row"].to_list()

    # every page has an original-row point (incl. unmatched ones); only matched
    # pages get a zip point
    assert rows.count("original") == review_df.height
    assert rows.count("mapped zip") == n_matched
    # unmatched pages have null zip text length in the review frame
    assert review_df.filter(~pl.col.is_matched)["zip_text_len"].null_count() == (
        review_df.height - n_matched
    )
    # in the plot data, unmatched pages must have a non-null match_score (0.0) so
    # the point still renders (a null size value would drop the mark), and are
    # flagged has_match=False so they draw as hollow rings
    plot_df = chart.data
    unmatched = plot_df.filter(~pl.col.has_match)
    assert unmatched.height >= 1
    assert unmatched["match_score"].null_count() == 0
    assert (unmatched["match_score"] == 0.0).all()

    # the points layer must encode a status-colored stroke so hollow (zero-fill)
    # points - e.g. unmatched original pages - still render as visible rings
    # rather than disappearing when their fill is transparent
    spec = chart.to_dict()
    points_layer = next(
        layer for layer in spec["layer"] if layer["mark"].get("type") == "point"
    )
    assert points_layer["encoding"]["stroke"]["field"] == "status"


def test_plot_alignment_x_domain_limited_to_plotted_pages(tmp_path):
    # excerpt-like: only a few corpus pages, but the zip spans the whole volume.
    # the x-axis domain should track the plotted pages, not the full zip range.
    seeds = [f"chapter-{i}-unique-content" for i in range(3)]
    pages = [
        {"id": "work.00000100", "order": 100, "text": _long_text(seeds[0])},
        {"id": "work.00000101", "order": 101, "text": _long_text(seeds[1])},
        {"id": "work.00000250", "order": 250, "text": _long_text(seeds[2])},
    ]
    zip_files = {f"{o:08d}.txt": "junk page" for o in range(1, 301)}
    zip_files["00000100.txt"] = _long_text(seeds[0])
    zip_files["00000101.txt"] = _long_text(seeds[1])
    zip_files["00000250.txt"] = _long_text(seeds[2])
    review_df = review_alignment("work", pages, make_zip(tmp_path, zip_files))

    chart = plot_alignment(review_df)
    spec = chart.to_dict()

    # x domain should be near [100, 250] (plotted pages), not [1, 300] (full zip)
    for layer in spec["layer"]:
        domain = layer["encoding"]["x"]["scale"]["domain"]
        assert domain[0] >= 95
        assert domain[1] <= 255


# --- main / --continue ---


@pytest.fixture(autouse=True)
def _restore_signal_state():
    """main() installs SIGINT/SIGTERM handlers and toggles a module flag;
    restore both after each test so handlers don't leak across the suite."""
    orig_int = signal.getsignal(signal.SIGINT)
    orig_term = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGINT, orig_int)
    signal.signal(signal.SIGTERM, orig_term)
    dataset_prep._stop_requested = False


def _pages_through(work_id, pages, image_dir, tar, ht1930_work_ids=None):
    """Stand-in for process_work that yields pages unchanged (no images)."""
    yield from pages


def _write_corpus(path: Path, page_records: list[dict]) -> None:
    """Write a list of page dicts to a JSONL corpus file."""
    orjsonl.save(path, page_records)


def _write_metadata(corpus_dir: Path, work_ids: list[str]) -> None:
    """Write a minimal ppa_metadata.csv with the columns get_ht1930_work_ids
    needs (work_id, pub_year, source, pages_digital) for each work id."""
    lines = ["work_id,pub_year,source,pages_digital"]
    lines += [f"{work_id},1850,Gale," for work_id in work_ids]
    (corpus_dir / "ppa_metadata.csv").write_text("\n".join(lines) + "\n")


def _make_corpus_dir(
    corpus_dir: Path, page_records: list[dict], pages_filename="ppa_pages.jsonl"
) -> Path:
    """Create a PPA corpus directory with a page corpus and work metadata."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    _write_corpus(corpus_dir / pages_filename, page_records)
    work_ids = sorted({p["work_id"] for p in page_records})
    _write_metadata(corpus_dir, work_ids)
    return corpus_dir


def _run_main(corpus_dir, image_dir, output_dir, extra_args=None):
    """Invoke main() with the given positional args (+ optional extras),
    patching process_work so no image/zip handling is exercised."""
    argv = [
        "dataset_prep.py",
        str(corpus_dir),
        str(image_dir),
        str(output_dir),
    ]
    if extra_args:
        argv += extra_args
    with (
        patch("sys.argv", argv),
        patch(
            "corppa.utils.dataset_prep.process_work",
            side_effect=_pages_through,
        ),
    ):
        main()


@pytest.fixture
def corpus_input(tmp_path):
    """A small two-work corpus (dir with pages + metadata), two pages each."""
    return _make_corpus_dir(
        tmp_path / "corpus",
        [
            {"work_id": "workA", "id": "workA.0001", "text": "a1"},
            {"work_id": "workA", "id": "workA.0002", "text": "a2"},
            {"work_id": "workB", "id": "workB.0001", "text": "b1"},
            {"work_id": "workB", "id": "workB.0002", "text": "b2"},
        ],
    )


def test_main_progress_bar_enabled_by_default(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    with patch("corppa.utils.dataset_prep.tqdm", wraps=tqdm) as mock_tqdm:
        _run_main(corpus_input, image_dir, output_dir)

    # progress bar is shown (not disabled) unless --no-progress is passed
    assert mock_tqdm.call_args.kwargs["disable"] is False


def test_main_no_progress_disables_bar(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    with patch("corppa.utils.dataset_prep.tqdm", wraps=tqdm) as mock_tqdm:
        _run_main(corpus_input, image_dir, output_dir, extra_args=["--no-progress"])

    assert mock_tqdm.call_args.kwargs["disable"] is True


def test_main_writes_all_works(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    _run_main(corpus_input, image_dir, output_dir)

    output_pages = output_dir / "ppa_pages.jsonl"
    output_tar = output_dir / "ppa_images.tar"
    assert output_pages.exists()
    # tar is uncompressed (not .tar.gz) so it can be appended to on continue
    assert output_tar.exists()
    assert not (output_dir / "ppa_images.tar.gz").exists()

    written = list(orjsonl.stream(output_pages))
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]


def test_main_continue_skips_completed_works(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_pages = output_dir / "ppa_pages.jsonl"
    output_tar = output_dir / "ppa_images.tar"
    # simulate a previous run that already completed workA
    _write_corpus(
        output_pages,
        [
            {"work_id": "workA", "id": "workA.0001", "text": "a1"},
            {"work_id": "workA", "id": "workA.0002", "text": "a2"},
        ],
    )
    # and produced an existing (uncompressed) tar
    with tarfile.open(output_tar, "w"):
        pass

    _run_main(corpus_input, image_dir, output_dir, extra_args=["--continue"])

    written = list(orjsonl.stream(output_pages))
    # workA pages are preserved and only appear once; workB is appended
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]


def test_main_continue_skips_completed_last_work(tmp_path, corpus_input, caplog):
    # when the LAST work in the corpus is already completed, the end-of-loop
    # handler must count it as skipped (not reprocess it)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_pages = output_dir / "ppa_pages.jsonl"
    output_tar = output_dir / "ppa_images.tar"
    # simulate a previous run that already completed workB (the last work)
    _write_corpus(
        output_pages,
        [
            {"work_id": "workB", "id": "workB.0001", "text": "b1"},
            {"work_id": "workB", "id": "workB.0002", "text": "b2"},
        ],
    )
    with tarfile.open(output_tar, "w"):
        pass

    with caplog.at_level("INFO", logger="corppa.utils.dataset_prep"):
        _run_main(corpus_input, image_dir, output_dir, extra_args=["--continue"])

    # workA is appended; workB (already present, and the last work) is not
    # duplicated
    written = list(orjsonl.stream(output_pages))
    assert [p["id"] for p in written] == [
        "workB.0001",
        "workB.0002",
        "workA.0001",
        "workA.0002",
    ]
    # the last work being skipped is reflected in the summary counts
    assert (
        "finished: 1 works processed (2 pages, 0 page images), "
        "1 works skipped (2 pages)" in caplog.text
    )


def test_main_continue_does_not_rename_existing_output(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_pages = output_dir / "ppa_pages.jsonl"
    _write_corpus(
        output_pages,
        [{"work_id": "workA", "id": "workA.0001", "text": "a1"}],
    )

    _run_main(corpus_input, image_dir, output_dir, extra_args=["--continue"])

    # continue appends in place; it must not create a .bak backup
    assert not (output_dir / "ppa_pages.jsonl.bak").exists()


def test_main_continue_missing_output_starts_fresh(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    # --continue with no existing output should behave like a fresh run
    _run_main(corpus_input, image_dir, output_dir, extra_args=["--continue"])

    written = list(orjsonl.stream(output_dir / "ppa_pages.jsonl"))
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]


def test_main_without_continue_renames_existing_output(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_pages = output_dir / "ppa_pages.jsonl"
    _write_corpus(
        output_pages,
        [{"work_id": "old", "id": "old.0001", "text": "old"}],
    )

    _run_main(corpus_input, image_dir, output_dir)

    # existing output is renamed to a .bak file and rewritten fresh
    backup = output_dir / "ppa_pages.jsonl.bak"
    assert backup.exists()
    assert [p["id"] for p in orjsonl.stream(backup)] == ["old.0001"]
    written = list(orjsonl.stream(output_pages))
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]


def test_main_without_continue_warns_and_overwrites_existing_archive(
    tmp_path, corpus_input, caplog
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_tar = output_dir / "ppa_images.tar"
    # a leftover archive from a prior run, with a stale member to prove it is
    # overwritten (mode "w") rather than appended to
    with tarfile.open(output_tar, "w") as tar:
        info = tarfile.TarInfo(name="stale.txt")
        info.size = 0
        tar.addfile(info)

    with caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"):
        _run_main(corpus_input, image_dir, output_dir)

    # existing archive is flagged and overwritten (no stale member remains)
    assert "already exists, overwriting" in caplog.text
    with tarfile.open(output_tar, "r") as tar:
        assert "stale.txt" not in tar.getnames()


# --- graceful stop on signal ---


def test_main_stops_cleanly_after_current_work(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    # simulate a signal arriving while the first work is being processed:
    # request a stop after workA is handled, so workB is never started
    def stop_after_first(work_id, pages, image_dir, tar, ht1930_work_ids=None):
        if work_id == "workA":
            dataset_prep._request_stop(signal.SIGTERM, None)
        yield from pages

    argv = ["dataset_prep.py", str(corpus_input), str(image_dir), str(output_dir)]
    with (
        patch("sys.argv", argv),
        patch(
            "corppa.utils.dataset_prep.process_work",
            side_effect=stop_after_first,
        ),
    ):
        main()

    # only the completed work (workA) is written; workB is skipped entirely
    written = list(orjsonl.stream(output_dir / "ppa_pages.jsonl"))
    assert [p["id"] for p in written] == ["workA.0001", "workA.0002"]


def test_main_input_stream_close_error_is_suppressed(tmp_path, corpus_input, caplog):
    # Regression: for compressed input, orjsonl.stream decompresses via an xopen
    # subprocess. On ctrl-c the SIGINT kills that subprocess (exit code -2), so
    # tearing down the stream generator raises a spurious OSError/BrokenPipeError.
    # main() must stop cleanly and suppress that shutdown-only error rather than
    # let it propagate (previously surfaced as "BrokenPipeError: [Errno 32]").
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    real_pages = list(orjsonl.stream(corpus_input / "ppa_pages.jsonl"))

    def failing_stream(_path):
        """Yield pages, but when the generator is closed early (as happens when
        the main loop breaks on a stop signal), raise on teardown to mimic the
        killed decompressor subprocess being closed after a ctrl-c."""
        try:
            yield from real_pages
        except GeneratorExit:
            # close() throws GeneratorExit in; the real bug is the xopen
            # subprocess close raising during this teardown
            raise OSError("b'' (exit code -2)")

    # request a stop while processing the first work so the loop breaks mid-stream
    # and main() closes the (still-open) generator, triggering the teardown error
    def stop_after_first(work_id, pages, image_dir, tar, ht1930_work_ids=None):
        if work_id == "workA":
            dataset_prep._request_stop(signal.SIGINT, None)
        yield from pages

    argv = ["dataset_prep.py", str(corpus_input), str(image_dir), str(output_dir)]
    with (
        patch("sys.argv", argv),
        patch(
            "corppa.utils.dataset_prep.process_work",
            side_effect=stop_after_first,
        ),
        patch(
            "corppa.utils.dataset_prep.orjsonl.stream",
            side_effect=failing_stream,
        ),
        caplog.at_level("DEBUG", logger="corppa.utils.dataset_prep"),
    ):
        # must not raise despite the stream teardown error
        main()

    # the completed work (workA) was written before the clean stop
    written = list(orjsonl.stream(output_dir / "ppa_pages.jsonl"))
    assert [p["id"] for p in written] == ["workA.0001", "workA.0002"]
    assert "ignoring input stream close error during shutdown" in caplog.text


def test_main_stop_flag_reset_between_runs(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    # leave the module flag set from a prior run; main() should reset it so
    # this run completes normally
    dataset_prep._stop_requested = True

    _run_main(corpus_input, image_dir, output_dir)

    assert dataset_prep._stop_requested is False
    written = list(orjsonl.stream(output_dir / "ppa_pages.jsonl"))
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]


# --- run summary reporting ---


def test_main_reports_finished_counts(tmp_path, corpus_input, caplog):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    with caplog.at_level("INFO", logger="corppa.utils.dataset_prep"):
        _run_main(corpus_input, image_dir, output_dir)

    # two works, two pages each, no images added by the stand-in process_work
    assert (
        "finished: 2 works processed (4 pages, 0 page images), "
        "0 works skipped (0 pages)" in caplog.text
    )


def test_main_reports_page_image_counts(tmp_path, corpus_input, caplog):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    # simulate process_work adding an image path to one page per work
    def add_one_image(work_id, pages, image_dir, tar, ht1930_work_ids=None):
        for i, page in enumerate(pages):
            if i == 0:
                page["image_path"] = f"{work_id}/{page['id']}.jpg"
            yield page

    argv = ["dataset_prep.py", str(corpus_input), str(image_dir), str(output_dir)]
    with (
        patch("sys.argv", argv),
        patch(
            "corppa.utils.dataset_prep.process_work",
            side_effect=add_one_image,
        ),
        caplog.at_level("INFO", logger="corppa.utils.dataset_prep"),
    ):
        main()

    # one image per work = two page images across the two works
    assert (
        "finished: 2 works processed (4 pages, 2 page images), "
        "0 works skipped (0 pages)" in caplog.text
    )


def test_main_reports_skipped_counts_on_continue(tmp_path, corpus_input, caplog):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_pages = output_dir / "ppa_pages.jsonl"
    output_tar = output_dir / "ppa_images.tar"
    # simulate a previous run that already completed workA
    _write_corpus(
        output_pages,
        [
            {"work_id": "workA", "id": "workA.0001", "text": "a1"},
            {"work_id": "workA", "id": "workA.0002", "text": "a2"},
        ],
    )
    with tarfile.open(output_tar, "w"):
        pass

    with caplog.at_level("INFO", logger="corppa.utils.dataset_prep"):
        _run_main(corpus_input, image_dir, output_dir, extra_args=["--continue"])

    # workA is skipped (2 pages), workB is processed (2 pages)
    assert (
        "finished: 1 works processed (2 pages, 0 page images), "
        "1 works skipped (2 pages)" in caplog.text
    )


def test_main_reports_interrupted_counts(tmp_path, corpus_input, caplog):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    # request a stop after workA so workB is never processed
    def stop_after_first(work_id, pages, image_dir, tar, ht1930_work_ids=None):
        if work_id == "workA":
            dataset_prep._request_stop(signal.SIGTERM, None)
        yield from pages

    argv = ["dataset_prep.py", str(corpus_input), str(image_dir), str(output_dir)]
    with (
        patch("sys.argv", argv),
        patch(
            "corppa.utils.dataset_prep.process_work",
            side_effect=stop_after_first,
        ),
        caplog.at_level("INFO", logger="corppa.utils.dataset_prep"),
    ):
        main()

    # only workA is processed before the stop; report reads "interrupted"
    assert (
        "interrupted: 1 works processed (2 pages, 0 page images), "
        "0 works skipped (0 pages)" in caplog.text
    )
