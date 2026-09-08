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
    get_zip_textfiles,
    longest_increasing_subseq,
    main,
    process_gale_work,
    process_ht_work,
    process_work,
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
    assert "duplicate page filename" not in caplog.text


def test_align_shifted_pages_logs_unmatched_pages(caplog):
    # pages 1-4 shift +10 -> aligned orders 11,12,13,14, but the zip only has
    # 11,12,13; page 4 aligns to a missing zip page and gets no filename
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

    assert "1 of 4 page(s) did not align to a zip page filename" in caplog.text


def test_align_shifted_pages_non_monotonic_warns(caplog):
    # later original pages align to earlier zip pages: sorting by original order,
    # aligned_order goes backwards (11,12 then 3,4) -> should warn.
    seeds = [f"chapter-{i}-unique-content" for i in range(4)]
    pages_df, zip_pages_df = _make_shifted_frames(
        page_orders=[1, 2, 3, 4],
        zip_orders=[11, 12, 3, 4],
        seeds=seeds,
    )

    with caplog.at_level("WARNING", logger="corppa.utils.dataset_prep"):
        align_shifted_pages(pages_df, zip_pages_df)

    assert "aligned page order is not monotonic" in caplog.text


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


def _pages_through(work_id, pages, image_dir, tar):
    """Stand-in for process_work that yields pages unchanged (no images)."""
    yield from pages


def _write_corpus(path: Path, page_records: list[dict]) -> None:
    """Write a list of page dicts to a JSONL corpus file."""
    orjsonl.save(path, page_records)


def _run_main(input_path, image_dir, output_dir, extra_args=None):
    """Invoke main() with the given positional args (+ optional extras),
    patching process_work so no image/zip handling is exercised."""
    argv = [
        "dataset_prep.py",
        str(input_path),
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
    """A small two-work corpus with two pages each."""
    input_path = tmp_path / "input.jsonl"
    _write_corpus(
        input_path,
        [
            {"work_id": "workA", "id": "workA.0001", "text": "a1"},
            {"work_id": "workA", "id": "workA.0002", "text": "a2"},
            {"work_id": "workB", "id": "workB.0001", "text": "b1"},
            {"work_id": "workB", "id": "workB.0002", "text": "b2"},
        ],
    )
    return input_path


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
    def stop_after_first(work_id, pages, image_dir, tar):
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
    def add_one_image(work_id, pages, image_dir, tar):
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
    def stop_after_first(work_id, pages, image_dir, tar):
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
