# prep ppa text+image dataset for publication
import argparse
import logging
import signal
import tarfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from time import mktime, perf_counter
from typing import Optional
from zipfile import ZipFile

import numpy as np
import orjsonl
import polars as pl
import polars_ds as pds
import rapidfuzz
from intspan import intspan
from tqdm import tqdm

from corppa.utils.path_utils import (
    encode_htid,
    get_gale_image_name,
    get_ppa_source,
    get_vol_dir,
    get_volume_id,
)

logger = logging.getLogger(__name__)

# set when an interrupt/termination signal is received so the main loop can
# stop cleanly at the next work boundary (avoids partial-work output)
_stop_requested = False


def _request_stop(signum, frame):
    """Signal handler: request a clean stop after the current work finishes."""
    global _stop_requested
    logger.warning(
        "received %s; will stop after current work finishes",
        signal.Signals(signum).name,
    )
    _stop_requested = True


def get_zip_textfiles(zipfile: ZipFile) -> Iterator[tuple[str, str]]:
    """Return a generator of text files from an open zip archive. Returns tuples of (file_id, content)
    where file_id is the stem of the filename."""
    txtfile_list = [fn for fn in zipfile.namelist() if fn.endswith(".txt")]
    for filename in txtfile_list:
        with zipfile.open(filename) as txtfile:
            # path stem is the filename without the extension; return file id + contents
            yield (Path(filename).stem, txtfile.read().decode("utf-8"))


def get_zipfile_pages(zipfile) -> pl.DataFrame:
    """Load text files from a HathiTrust zipfile into a polars dataframe.
    Returned dataframe has the following columns:
        - page_filename
        - text : full text contents of the page
        - page_id : numeric portion extracted from page_filename
        - order : numeric version of `page_id`
        - text_len : number of characters in `text`
    """
    return (
        pl.DataFrame(
            data=get_zip_textfiles(zipfile),
            schema=["page_filename", "text"],
        )
        .with_columns(
            # extract the numeric page id for joining with page data
            # some works have filenames like OSU_32435051461309_00000602 ; others are simply numeric
            page_id=pl.col.page_filename.str.extract(r"_?([0-9]+$)")
        )
        .with_columns(
            # make an order field to match page id so we can calculate size of shift
            order=pl.col.page_id.cast(pl.Int64),
            text_len=pl.col.text.str.len_chars(),
        )
    )


def add_zip_file_to_tar(
    zipfile: ZipFile,
    zip_filename: str,
    tar: tarfile.TarFile,
    tar_file_path: str,
) -> None:
    """Add a single file from an open ZipFile to an open TarFile without
    extracting to disk. tar_file_path sets the path within the tar archive."""
    zipinfo = zipfile.getinfo(zip_filename)
    # create tar info object for destination path with size and modification time
    tarinfo = tarfile.TarInfo(name=tar_file_path)
    tarinfo.size = zipinfo.file_size
    # convert zip info modification time to tar info mtime
    tarinfo.mtime = mktime(zipinfo.date_time + (0, 0, -1))  # convert to timestamp
    with zipfile.open(zipinfo) as f:
        tar.addfile(tarinfo, fileobj=f)


def get_zip_imgexts(zipfile: ZipFile) -> list[str]:
    """HathiTrust zip files include images in multiple formats; returns
    a list of all unique image extensions found in the zip file."""
    exts = set()
    for filename in zipfile.namelist():
        file_path = Path(filename)
        # compare case-insensitive but return actual case
        if file_path.suffix.lower() in [".tif", ".jpg", ".jpeg", ".jp2"]:
            exts.add(file_path.suffix)

    return list(exts)


# minimum text length (in characters) for a page to be matched against zip pages
MIN_MATCH_TEXT_LEN = 600
# minimum similarity (0-100) for a page match to be considered;
# uses rapidfuzz.fuzz.ratio, which returns normalized Indel similarity
MATCH_SCORE_CUTOFF = 85
# a page's best zip match must beat its runner-up by at least this many ratio
# points to be trusted; guards against near-ties from repeated boilerplate pages
MATCH_SCORE_MARGIN = 3
# a best match at or above this ratio is treated as an unambiguous match and
# trusted regardless of the runner-up margin (near-exact text match)
MATCH_SCORE_STRONG = 99


def align_shifted_pages(pages_df: pl.DataFrame, zip_pages_df: pl.DataFrame):
    """Align corpus pages to zip page filenames when page order has shifted
    between versions. Shifts are determined by matching pages with sufficient text
    in the first dataframe to pages in the zip file using normalized indel similarity
    (rapidfuzz.fuzz.ratio). Short pages and pages with low-confidence matches are aligned
    based on the shift of the nearest preceding alignment, or nearest following alignment
    if no preceding alignment.

    Returns a DataFrame with ``id`` and ``page_filename`` columns where each row
    is the determined alignment; returns an empty dataframe if alignment could not be determined.
    ."""
    # empty mapping frame, used as the early-return value when no long pages
    # clear the filter or no reliable shift can be determined
    empty_mapping_df = pl.DataFrame(
        schema={"id": pl.String, "page_filename": pl.String}
    )

    # sort by order and keep the full (unfiltered) set; short pages still need
    # to be mapped even though they don't contribute match evidence
    orig_pages_df = pages_df.sort("order", descending=False)

    # long pages only: these are the pages we trust to determine the shift
    long_pages_df = orig_pages_df.with_columns(
        text_len=pl.col.text.str.len_chars()
    ).filter(pl.col.text_len > MIN_MATCH_TEXT_LEN)
    if long_pages_df.height == 0:
        # no long-enough pages - can't determine a shift, so give up
        logger.warning(
            "No pages over %d characters; cannot determine page shift",
            MIN_MATCH_TEXT_LEN,
        )
        return empty_mapping_df

    # compare every long page in the original set with every page in the zip file
    # returns an (n_long, n_zip) matrix; scores below the cutoff are 0
    scores = rapidfuzz.process.cdist(
        long_pages_df["text"].to_list(),
        zip_pages_df["text"].to_list(),
        scorer=rapidfuzz.fuzz.ratio,
        workers=-1,
        score_cutoff=MATCH_SCORE_CUTOFF,
    )

    # get page order (digital sequence) for both sets as ndarray
    zip_orders = zip_pages_df["order"].to_numpy()
    long_orders = long_pages_df["order"].to_numpy()

    # determine best match for each long page, plus the next best to ensure high-confidence match
    # argpartition on the last two positions gathers each row's two largest
    # scores (unordered between themselves) in one pass; take their max/min to
    # get the best and runner-up, and the argpartition index for the best.
    row_idx = np.arange(scores.shape[0])
    if scores.shape[1] >= 2:
        top2_idx = np.argpartition(scores, -2, axis=1)[:, -2:]
        top2 = scores[row_idx[:, None], top2_idx]
        # the larger of the two is the best match; the other is the runner-up
        best_pos = top2.argmax(axis=1)
        best_idx = top2_idx[row_idx, best_pos]
        best_score = top2.max(axis=1)
        second_score = top2.min(axis=1)
    else:
        # only one zip page: no runner-up to compare against
        best_idx = scores.argmax(axis=1)
        best_score = scores[row_idx, best_idx]
        second_score = np.zeros_like(best_score)

    # determine match confidence for each long page based on either:
    # - strong match score
    # - good match score (above the cutoff) that is clearly better than the next match
    confident = (best_score > 0) & (
        (best_score >= MATCH_SCORE_STRONG)
        | ((best_score - second_score) >= MATCH_SCORE_MARGIN)
    )
    # shift is defined as zip order minus original order, so the aligned zip
    # order for a page is recovered by original order + shift
    shifts = zip_orders[best_idx] - long_orders
    trusted_shift = np.where(confident, shifts, np.nan)

    long_shift_df = long_pages_df.select("order").with_columns(
        # NaN marks pages without confident matches; convert to nulls so we can forward/back fill
        shift=pl.Series(trusted_shift).fill_nan(None)
    )
    if long_shift_df["shift"].drop_nulls().is_empty():
        # no long page produced a confident, unambiguous match
        logger.warning("No high-confidence matches found; cannot determine page shift")
        return empty_mapping_df

    # combine the high-confidence alignment shift values into the full page dataframe;
    # left join to keep all pages; shift is null for short pages & low-confidence matches
    pages_shift_df = orig_pages_df.join(
        long_shift_df, on="order", how="left"
    ).with_columns(
        # determine shift for all pages; use nearest high-confidence match (preceding page, then following)
        # to determine shift for pages without alignment
        inferred_shift=pl.col.shift.forward_fill().backward_fill()
    )
    # summarize the shift for logging output when info-level is enabled
    if logger.isEnabledFor(logging.INFO):
        shift_summary_df = pages_shift_df.group_by("inferred_shift").agg(
            n_pages=pl.len(), orders=pl.col.order
        )
        # count how many alignments were inferred
        num_inferred = pages_shift_df.filter(pl.col.shift.is_null()).height
        pct_inferred = f"{num_inferred / pages_df.height:.1%}"
        # use intspan to combine the list of pages into a readable format
        shift_summary = "; ".join(
            f"{int(row['inferred_shift']):+d} ({intspan(row['orders'])}, {row['n_pages']:,} pages)"
            for row in shift_summary_df.iter_rows(named=True)
        )
        logger.info(
            "page shift: %s \t%d alignment%s inferred (%s)",
            shift_summary,
            num_inferred,
            ""
            if num_inferred == 1
            else "s",  # conditionallypluralize inferred alignment
            pct_inferred,
        )

    # single join for the whole work: shift each page's order and look up the
    # zip page filename at the aligned order
    page_mapping_df = pages_shift_df.with_columns(
        # calculate the aligned order by applying the actual or inferred shift to original order
        aligned_order=(pl.col.order + pl.col.inferred_shift).cast(pl.Int64)
    ).join(
        # then join all pages on the new aligned order
        zip_pages_df.select(["order", "page_filename"]),
        left_on="aligned_order",
        right_on="order",
        how="left",
    )

    # sanity-check the alignment; warn (but don't fail) on anything suspicious so
    # a questionable mapping is surfaced without halting the whole run.
    matched = page_mapping_df.filter(pl.col.page_filename.is_not_null())
    # report pages that did not align to any zip page (no filename)
    num_unmatched = page_mapping_df.height - matched.height
    if num_unmatched:
        logger.info(
            "%d of %d page(s) did not align to a zip page filename",
            num_unmatched,
            page_mapping_df.height,
        )
    # two original pages should never map to the same zip page filename
    num_dupes = matched.height - matched["page_filename"].n_unique()
    if num_dupes:
        logger.warning("alignment produced %d duplicate page filename(s)", num_dupes)
    # aligned order should preserve original page order: sorting by original
    # order, aligned_order should be strictly increasing. Gaps are fine (pages
    # can be removed between versions); order going backwards is not.
    aligned = matched.sort("order")["aligned_order"]
    if aligned.len() > 1 and not (aligned.diff().drop_nulls() > 0).all():
        logger.warning("aligned page order is not monotonic (pages out of order)")

    return page_mapping_df.select(["id", "page_filename"])


# determine alignment between pages in different versions of hathitrust
def align_pages(work_id: str, pages_df: pl.DataFrame, zipfile: ZipFile) -> dict:
    expected_page_count = pages_df.height
    # load text files from zipfile into a polars dataframe
    zip_pages_df = get_zipfile_pages(zipfile)

    # NOTE: for excerpt, page count is not expected to match but should be >= total
    zip_count_mismatch = zip_pages_df.height < expected_page_count
    if zip_count_mismatch:
        logger.warning(
            "%s page count mismatch; pages in zipfiles (%d, expected at least %d)",
            work_id,
            zip_pages_df.height,
            expected_page_count,
        )
    # join origin pages with zip pages on the numeric page id,
    # and calculate a fuzzy text match score for each page using rapidfuzz fuzz ratio (normalized indel similarity)
    pages_join_df = (
        pages_df.with_columns(page_id=pl.col.id.str.extract(r"[._]([0-9]+$)"))
        .join(zip_pages_df, on="page_id")
        # NOTE: if any multiprocessing is added to this script, remove parallel=True argument
        .with_columns(text_match=pds.str_fuzz("text", "text_right", parallel=True))
    )
    # only warn about the joined page count if we didn't already warn about
    # the zip page count above, to avoid a redundant warning
    if not zip_count_mismatch and expected_page_count != pages_join_df.height:
        logger.warning(
            "%s joined pages (%d) does not match expected page count (%d)",
            work_id,
            pages_join_df.height,
            expected_page_count,
        )

    # determine the average score for pages with text (polars skips nulls in aggregation),
    # as a way to check the overall alignment between the two sets of pages
    avg = pages_join_df["text_match"].mean()
    logger.info(
        f"{work_id: <30} {pages_df.height:> 4,} pages; average indel similarity score: {avg:.3f}"
    )
    # at least one 0.87 is visibly correct alignment; use same cutoff as for the
    # shift alignment, but adjust for the 0-1 score rather than 1-100 like cdist
    if avg is not None and (avg * 100) > MATCH_SCORE_CUTOFF:
        page_mapping_df = pages_join_df
    else:
        page_mapping_df = align_shifted_pages(pages_df, zip_pages_df)
        if page_mapping_df.is_empty():
            return {}

    # construct and return a dictionary mapping original page id to corresponding filename in the zipfile
    return {
        r["id"]: r["page_filename"]
        for r in page_mapping_df.select(["id", "page_filename"]).iter_rows(named=True)
    }


def process_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    # generic process work method, which calls appropriate source-specific method
    source = get_ppa_source(work_id)
    match source:
        case "Gale":
            yield from process_gale_work(work_id, pages, image_dir, tar)
        case "HathiTrust":
            yield from process_ht_work(work_id, pages, image_dir, tar)
        case "EEBO-TCP":
            yield from pages  # no images
        case _:
            # unknown source: don't silently drop pages, yield them unchanged
            logger.warning(
                "unknown source %r for work %s; yielding pages without images",
                source,
                work_id,
            )
            yield from pages


def process_gale_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    vol_id = get_volume_id(work_id)
    vol_img_dir = image_dir / get_vol_dir(vol_id)
    if vol_img_dir.is_dir():
        logging.debug("%s : %s : %d pages", work_id, vol_img_dir, len(pages))
        for page in pages:
            # page id is vol id + sequence, e.g. CB0127060085.0005; use the
            # trailing sequence as the page number for the shared filename helper
            page_num = int(page["id"].rsplit(".", 1)[-1])
            image_path = vol_img_dir / get_gale_image_name(vol_id, page_num)
            if image_path.is_file():
                tar_image_path = f"{work_id}/{image_path.name}"
                tar.add(image_path, arcname=tar_image_path)
                # add the image path in the tar file to the page data
                page["image_path"] = tar_image_path
            # yield page data either way (with or without image path)
            yield page
    else:
        # no image directory for this volume; yield each page unchanged
        yield from pages


def process_ht_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    htid = get_volume_id(work_id)
    # zip file is named based on id without institution prefix
    # must be encoded to convert ark style ids to file safe format
    htid_suffix = encode_htid(htid).split(".")[-1]
    zipfile_path = image_dir / "HathiTrust" / encode_htid(htid) / f"{htid_suffix}.zip"
    if not zipfile_path.exists():
        # logger.warning("zipfile %s does not exist, omitting images", zipfile_path)
        # yield pages without image paths
        yield from pages
    else:
        with ZipFile(zipfile_path) as ht_zip:
            page_mapping = align_pages(work_id, pl.DataFrame(pages), ht_zip)
            if not page_mapping:
                logger.warning(
                    "no page mapping found for work %s, omitting images",
                    work_id,
                )
                # yield pages without image paths
                yield from pages
            else:
                # when image mapping was returned, add images to tar file and image paths to page data
                img_exts = get_zip_imgexts(ht_zip)
                for page in pages:
                    page_id = page["id"]  # .split(".")[-1]
                    # get the corresponding image from the zip, add to the tar file with appropriate name,
                    # and add the image path to the page record for output
                    page_basename = page_mapping.get(page_id)

                    # add the image from the corresponding path in the zipfile to the
                    # appropriate path for this page in the tarfile
                    file_namelist = ht_zip.namelist()
                    if page_basename is not None:
                        zip_image_basepath = f"{htid_suffix}/{page_basename}"
                        for img_ext in img_exts:
                            zip_image_path = f"{zip_image_basepath}{img_ext}"
                            if zip_image_path in file_namelist:
                                break
                        tar_image_path = f"{encode_htid(htid)}/{page_id}{img_ext}"
                        try:
                            add_zip_file_to_tar(
                                ht_zip, zip_image_path, tar, tar_image_path
                            )
                            # if adding succeeded, add the image path in the page record for output
                            page["image_path"] = tar_image_path

                        except KeyError:
                            has_text = page["text"].strip() != ""
                            if has_text:
                                logger.warning(
                                    "image %s not found in zipfile but page has text; skipping",
                                    zip_image_path,
                                )
                            logger.debug(
                                "matching filenames: %s",
                                [f for f in file_namelist if page_basename in f],
                            )

                    # yield every page whether or not an image was aligned/added,
                    # so no pages are dropped from the output corpus
                    yield page


def main():
    global _stop_requested
    _stop_requested = False

    parser = argparse.ArgumentParser(
        description="Prepare PPA full-text dataset for publication by aligning pages and organizing images",
    )
    parser.add_argument(
        "input",
        help="PPA full-text corpus; must be a JSONL file (compressed or not)",
        type=Path,
    )
    parser.add_argument(
        "image_dir",
        help="Base directory for images",
        type=Path,
    )
    parser.add_argument(
        "output_dir",
        help="Directory where the updated page corpus and image archive file should be saved",
        type=Path,
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue a previous run: append to existing output files and "
        "skip works already present in the output JSONL",
    )
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-level",
        default="info",
        type=str.lower,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity (default: info); case-insensitive",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write log output to this file instead of stderr",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)s: %(message)s",
        filename=args.log_file,
    )

    if not args.output_dir.is_dir():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    output_pages_path = (
        args.output_dir / "ppa_pages.jsonl"
    )  # .gz # disable compression for now, for testing
    # uncompressed tar so it can be reopened in append mode when continuing
    output_archive_path = args.output_dir / "ppa_images.tar"

    # set of work ids already present in the output; populated when continuing
    completed_work_ids: set[str] = set()

    if args.continue_run:
        # append to existing output; collect work ids already written so we can
        # skip them, rather than renaming/overwriting the existing files
        if output_pages_path.exists():
            completed_work_ids = set(
                pl.scan_ndjson(output_pages_path)
                .select("work_id")
                .unique()
                .collect()
                .get_column("work_id")
            )
            logger.info(
                "Adding to existing output: %s works already in %s",
                f"{len(completed_work_ids):,}",
                output_pages_path,
            )
        else:
            logger.warning(
                "--continue set but output file %s does not exist; starting fresh",
                output_pages_path,
            )
        # append to the tar if it exists, otherwise create it
        tar_mode = "a" if output_archive_path.exists() else "w"
    else:
        tar_mode = "w"
        if output_pages_path.exists():
            # because we extend, we need to rename any existing output file
            old_output_pages = output_pages_path.with_suffix(".jsonl.bak")
            output_pages_path.rename(old_output_pages)
            logger.warning(
                "output file %s exists; renamed to %s",
                output_pages_path,
                old_output_pages,
            )
        if output_archive_path.exists():
            logger.warning(
                "output file %s already exists, overwriting", output_archive_path
            )

    # use a polars lazy frame to calculate the total so tqdm can estimate completion
    start_time = perf_counter()
    total_pages = pl.scan_ndjson(args.input).select(pl.len()).collect().item()
    end_time = perf_counter()
    logger.info(
        "%s total pages (calculated in %0.2fs)",
        f"{total_pages:,}",
        end_time - start_time,
    )
    # configure tqdm to format as comma delimited numbers - from https://stackoverflow.com/a/76964589
    tqdm.format_sizeof = lambda x, divisor=None: f"{x:,}" if divisor else f"{x:5.2f}"
    # stop cleanly at the next work boundary on ctrl-c (SIGINT) or termination
    # (SIGTERM, e.g. SLURM timeout); output always ends on a whole work so a
    # --continue run can resume from the first unwritten work
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    # Stream pages one at a time; corpus is sorted by work+page so we can
    # process pages by work as the work_id changes.
    # tally works, pages, and page images handled so we can report totals when
    # the run finishes or is interrupted
    counts: defaultdict[str, int] = defaultdict(int)
    with tarfile.open(output_archive_path, tar_mode) as tar:
        prev_work_id: Optional[str] = None
        pages: list[dict] = []
        # whether the current work should be skipped (already in output)
        skip_work = False
        for page in tqdm(
            orjsonl.stream(args.input),
            desc="Reading pages",
            total=total_pages,
            unit_scale=True,
            disable=not args.progress,
        ):
            work_id = page["work_id"]
            # when work id changes, process the previous work pages and reset for the next
            if work_id != prev_work_id:
                if prev_work_id is not None:
                    if skip_work:
                        counts["works_skipped"] += 1
                    else:
                        pages = list(
                            process_work(prev_work_id, pages, args.image_dir, tar)
                        )
                        orjsonl.extend(output_pages_path, pages)
                        counts["works_processed"] += 1
                        counts["pages_processed"] += len(pages)
                        counts["page_images"] += sum(
                            1 for p in pages if p.get("image_path")
                        )
                # stop here (at a work boundary) if a signal was received, so we
                # never interrupt a work's tar/jsonl writes partway through; the
                # tar is still closed cleanly by the context manager
                if _stop_requested:
                    logger.warning("stopping cleanly after work %s", prev_work_id)
                    break
                prev_work_id = work_id
                pages = []
                # skip this work if it is already present in the output
                skip_work = work_id in completed_work_ids
            if skip_work:
                counts["pages_skipped"] += 1
            else:
                pages.append(page)

        # handle the pages for the last work at end of loop, unless we broke out
        # early on a stop signal (that work was already written before the break)
        if prev_work_id is not None and not _stop_requested:
            if skip_work:
                counts["works_skipped"] += 1
            else:
                pages = list(process_work(prev_work_id, pages, args.image_dir, tar))
                orjsonl.extend(output_pages_path, pages)
                counts["works_processed"] += 1
                counts["pages_processed"] += len(pages)
                counts["page_images"] += sum(1 for p in pages if p.get("image_path"))

    # report totals whether the run finished normally or stopped early
    logger.info(
        "%s: %s works processed (%s pages, %s page images), "
        "%s works skipped (%s pages)",
        "interrupted" if _stop_requested else "finished",
        f"{counts['works_processed']:,}",
        f"{counts['pages_processed']:,}",
        f"{counts['page_images']:,}",
        f"{counts['works_skipped']:,}",
        f"{counts['pages_skipped']:,}",
    )


if __name__ == "__main__":
    main()
