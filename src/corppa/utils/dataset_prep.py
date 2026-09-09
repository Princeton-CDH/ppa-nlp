# prep ppa text+image dataset for publication
import argparse
import bisect
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
        .sort(
            "order"
        )  # ensure pages are sorted in logical order, since zipfile does not guarantee this
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
# max characters of page text to include in the detailed-frame hover snippet
TEXT_SNIPPET_LEN = 140


def _text_snippet_expr(col: str) -> pl.Expr:
    """Polars expression for a short hover-friendly snippet of a text column:
    all whitespace (including newlines) collapsed to single spaces, then
    truncated to ``TEXT_SNIPPET_LEN`` characters (with an ellipsis when
    truncated). Newlines are ignored so the snippet reads as one line."""
    consolidated = (
        pl.col(col)
        .str.replace_all(r"\s+", " ")  # collapse all whitespace (incl. newlines)
        .str.strip_chars()
    )
    return (
        pl.when(consolidated.str.len_chars() > TEXT_SNIPPET_LEN)
        .then(consolidated.str.slice(0, TEXT_SNIPPET_LEN) + pl.lit("…"))
        .otherwise(consolidated)
    )


def longest_increasing_subseq(values: np.ndarray) -> np.ndarray:
    """Return the indices (into ``values``) of the longest *strictly* increasing
    subsequence. When several subsequences tie for longest, one is returned
    deterministically (same input always yields the same result); which of the
    tied subsequences is chosen is not part of the contract, since downstream
    alignment depends only on the derived shifts and the rapidfuzz-based dedup,
    not on the specific anchor set.

    Used by `align_shifted_pages` to filter high-confidence mappings down to a
    set of "anchor" matches that enforce sequential alignment while allowing for
    gaps on either side.

    We use this to reduce a set of candidate page matches down to a monotonic,
    conflict-free set of "anchors". Each confident match pairs an original page
    with a zip page (identified by its ``order``, the digital page sequence).
    Reading those zip page orders in original-page order, a valid alignment must
    be strictly increasing: later original pages can only map to later zip pages,
    and each zip page may be claimed at most once (strict increase => no repeats).
    Any confident match that would break that ordering - e.g. repeated boilerplate
    pages that all matched the same zip page, or a match that jumps backwards - is
    dropped by keeping only the longest increasing run.

    This is the classic O(n log n) patience-sorting algorithm.
    """
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=int)

    # tails_idx[k] holds the index (into `values`) of the smallest possible tail
    # value for an increasing subsequence of length k+1 found so far.
    tails_idx: list[int] = []
    # tail_values mirrors tails_idx but holds the values themselves, kept in a
    # separate list so it stays strictly increasing and can be binary-searched
    # directly (avoids rebuilding it each iteration, keeping this O(n log n)).
    tail_values: list = []
    # prev[i] links element i back to the element that precedes it in the best
    # subsequence ending at i, so we can reconstruct the chain at the end.
    prev = [-1] * n

    for i, v in enumerate(values):
        # find the first existing tail that is >= v; bisect_left (not _right)
        # makes this *strict*, so an equal value replaces rather than extends -
        # this is what prevents a zip page from being claimed twice.
        pos = bisect.bisect_left(tail_values, v)

        if pos == len(tails_idx):
            # v is larger than every current tail: it extends the longest run
            tails_idx.append(i)
            tail_values.append(v)
        else:
            # v can start/continue a length-(pos+1) run with a smaller tail;
            # replacing keeps future values more likely to extend it
            tails_idx[pos] = i
            tail_values[pos] = v

        # whatever run v lands on, its predecessor is the tail of the run one
        # shorter (or none, if v starts a length-1 run)
        prev[i] = tails_idx[pos - 1] if pos > 0 else -1

    # reconstruct the chain by walking prev[] backwards from the last tail, which
    # is the end of a longest increasing subsequence
    result: list[int] = []
    k = tails_idx[-1]
    while k != -1:
        result.append(k)
        k = prev[k]
    # walked back-to-front, so reverse to restore original order
    return np.array(result[::-1], dtype=int)


def align_shifted_pages(
    pages_df: pl.DataFrame, zip_pages_df: pl.DataFrame, detailed: bool = False
):
    """Align corpus pages to zip page filenames when page order has shifted
    between versions. Shifts are determined by matching pages with sufficient text
    in the first dataframe to pages in the zip file using normalized indel similarity
    (rapidfuzz.fuzz.ratio). Short pages and pages with low-confidence matches are aligned
    based on the shift of the nearest preceding alignment, or nearest following alignment
    if no preceding alignment.

    Returns a DataFrame with ``id`` and ``page_filename`` columns where each row
    is the determined alignment; returns an empty dataframe if alignment could not be determined.

    When ``detailed`` is True, the returned frame instead includes the per-page
    diagnostic columns useful for reviewing/visualizing an alignment:
    ``id``, ``order`` (original page order), ``aligned_order`` (mapped zip order),
    ``page_filename``, ``cdist_best_score`` (best rapidfuzz.cdist score, 0-100,
    for every scored/long page; null for short pages), ``shift`` (trusted anchor
    shift, null for filled pages), ``inferred_shift`` (shift actually applied),
    and the raw ``text`` / ``zip_text`` for each page. Derived review fields
    (``is_anchor``/``is_matched`` flags, text lengths, snippets, extra scoring)
    are added by ``review_alignment`` rather than here.
    """
    # empty mapping frame, used as the early-return value when no long pages
    # clear the filter or no reliable shift can be determined. detailed callers
    # get the diagnostic schema; the pipeline gets the minimal id/filename schema.
    # Columns returned when detailed=True. This frame carries the raw alignment
    # data plus the original and matched-zip page text, so callers (e.g.
    # review_alignment) can derive any additional review fields themselves.
    detailed_schema = {
        "id": pl.String,
        "order": pl.Int64,
        "aligned_order": pl.Int64,
        "page_filename": pl.String,
        # best rapidfuzz.cdist score (0-100) for every scored (long) page against
        # any zip page; null for short pages that were not scored
        "cdist_best_score": pl.Float64,
        "shift": pl.Float64,
        "inferred_shift": pl.Float64,
        # raw text for the original page and its matched zip page (null if
        # unmatched); review/visualization helpers derive lengths, snippets, etc.
        "text": pl.String,
        "zip_text": pl.String,
    }
    empty_mapping_df = pl.DataFrame(
        schema=detailed_schema
        if detailed
        else {"id": pl.String, "page_filename": pl.String}
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
    if not confident.any():
        # bail out if no long page produced a confident, unambiguous match
        logger.warning("No high-confidence matches found; cannot determine page shift")
        return empty_mapping_df

    # Treat confident as candidate anchors, then filter to a strictly increasing sequence
    # to remove duplicate and out-of-order mappings.
    conf_pos = np.flatnonzero(confident)  # row indices of high-confidence matches
    conf_zip_cols = best_idx[conf_pos]  # matrix columns for high-confidence matches

    # Now filter zip page orders for high-confidence matches to strictly-increasing subset;
    # this results in a set of anchor mappings that follow sequence
    anchors = longest_increasing_subseq(zip_orders[conf_zip_cols])
    trusted_pos = conf_pos[anchors]

    # shift = zip order - original order
    # aligned order = original order + shift.
    # Use high-confidence sequential anchors to calculate an array of trusted shifts
    trusted_shift = np.full(scores.shape[0], np.nan)
    trusted_shift[trusted_pos] = (
        zip_orders[best_idx[trusted_pos]] - long_orders[trusted_pos]
    )
    long_shift_df = long_pages_df.select("order").with_columns(
        # shift for pages without trusted shift is NaN; convert to null to set up forward/back fill
        shift=pl.Series(trusted_shift).fill_nan(None),
        # best cdist score (0-100) for every long page; short pages (not scored)
        # get null via the left join below
        cdist_best_score=pl.Series(best_score, dtype=pl.Float64),
    )

    # pull the trusted shift shift values into the full page dataframe;
    # use a left join to keep all pages; shift is null for all but high-confidence sequential anchor pages.
    # maintain_order="left" is required: the forward/back-fill below propagates
    # each page's shift from its nearest sorted neighbor, so rows must stay in
    # original page order. Polars does not guarantee join row order otherwise.
    pages_shift_df = orig_pages_df.join(
        long_shift_df, on="order", how="left", maintain_order="left"
    ).with_columns(
        # determine shift for all pages; use nearest high-confidence match (preceding page, then following)
        # to determine shift for pages without alignment
        inferred_shift=pl.col.shift.forward_fill().backward_fill()
    )
    # single join for the whole work: shift each page's order and look up the
    # zip page filename at the aligned order. Pull the zip page text along too
    # (as zip_text) so we can score each mapping and break duplicate claims below.
    page_mapping_df = pages_shift_df.with_columns(
        # calculate the aligned order by applying the actual or inferred shift to original order
        aligned_order=(pl.col.order + pl.col.inferred_shift).cast(pl.Int64)
    ).join(
        # then join all pages on the new aligned order
        zip_pages_df.select(["order", "page_filename", pl.col.text.alias("zip_text")]),
        left_on="aligned_order",
        right_on="order",
        how="left",
    )

    # Check for and resolve any duplicate file mappings.
    # The forward/back-fill based on anchor shifts can make result in two adjacent
    # pages mapping to the *same* aligned zip page at a shift boundary,
    # since LIS only dedupes anchors.
    # If there are any non-null duplicate filenames, use rapidfuzz to choose
    # which page to match and which page to leave unmatched.
    is_dupe = pl.col.page_filename.is_not_null() & pl.col.page_filename.is_duplicated()
    if page_mapping_df.select(is_dupe.any()).item():
        page_mapping_df = page_mapping_df.with_columns(
            # score only the duplicate claimants against their zip page; unique
            # matches keep their filename regardless, so they need no score.
            # parallel=False: keep single-threaded (see align_pages note).
            match_score=pl.when(is_dupe)
            .then(pds.str_fuzz("text", "zip_text", parallel=False))
            .otherwise(None)
        ).with_columns(
            # Among the pages claiming the same filename, keep the best-scoring
            # (order as deterministic tiebreak) and clear the filename on the rest.
            # rank("ordinal") must be ascending (rank 1 = smallest); negate
            # match_score to make the *highest* score rank 1; order is already ascending
            # so the earliest page wins ties.
            page_filename=pl.when(
                ~is_dupe
                | (
                    pl.struct(score=-pl.col.match_score, order=pl.col.order)
                    .rank("ordinal")
                    .over("page_filename")
                    == 1
                )
            )
            .then(pl.col.page_filename)
            .otherwise(None)
        )

    # Summarize the shift for logging output when info-level is enabled.
    # A non-null page_filename means aligned_order joined to a real zip page, so
    # filtering on it reports the mapped (zip) page range that actually matched
    # and excludes raw aligned_orders that fall outside the zip range (e.g.
    # negative) for over-extended inferred shifts or pages dropped by dedup above.
    if logger.isEnabledFor(logging.INFO):
        shift_summary_df = (
            # only pages that joined to a real zip page contribute a mapped order
            page_mapping_df.filter(pl.col.page_filename.is_not_null())
            .group_by("inferred_shift")
            .agg(
                n_pages=pl.len(),
                orders=pl.col.order,
                mapped_orders=pl.col.aligned_order,
                # first original page in each group, for sorting
                first_order=pl.col.order.min(),
            )
            # sort by first page so groups will be output in sequential page order
            .sort("first_order")
        )
        # count how many alignments were inferred (inferred shift but no shift)
        num_inferred = page_mapping_df.filter(pl.col.shift.is_null()).height
        pct_inferred = f"{num_inferred / pages_df.height:.1%}"
        num_unmatched = page_mapping_df.filter(pl.col.page_filename.is_null()).height
        # use intspan to combine the list of pages into a readable format;
        # report both the original page range and the mapped (zip) page range
        shift_summary = "; ".join(
            f"{int(row['inferred_shift']):+d} "
            f"({intspan(row['orders'])}:{intspan(row['mapped_orders'])}; "
            f"{row['n_pages']:,} pages)"
            for row in shift_summary_df.iter_rows(named=True)
        )
        logger.info(
            "page shift: %s \t%d alignment%s inferred (%s); %d page%s unmatched",
            shift_summary,
            num_inferred,
            # conditionally pluralize inferred alignment
            "" if num_inferred == 1 else "s",
            pct_inferred,
            num_unmatched,
            # conditionally pluralize number of unmatched pages
            "" if num_unmatched == 1 else "s",
        )

    # sanity-check the alignment; warn (but don't fail) on anything suspicious so
    # a questionable mapping is surfaced without halting the whole run.
    # (unmatched page count is already reported in the shift summary above;
    # duplicate filenames are impossible because of the dedup step above)
    matched = page_mapping_df.filter(pl.col.page_filename.is_not_null())
    # aligned order should preserve original page order: sorting by original
    # order, aligned_order should be strictly increasing. Gaps are fine (pages
    # can be removed between versions); order going backwards is not.
    aligned = matched.sort("order")["aligned_order"]
    if aligned.len() > 1 and not (aligned.diff().drop_nulls() > 0).all():
        logger.warning("aligned page order is not monotonic (pages out of order)")

    if detailed:
        # return the raw alignment frame (incl. page text) for review/visualization;
        # derived fields (anchor/matched flags, lengths, snippets, extra scoring)
        # are left to the caller
        return page_mapping_df.select(list(detailed_schema.keys())).sort("order")

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
        f"{work_id: <30} {pages_df.height:> 5,} pages; average indel similarity score: {avg:.3f}"
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


def review_alignment(
    work_id: str,
    pages: pl.DataFrame | list[dict],
    zipfile: ZipFile | Path | str,
) -> pl.DataFrame:
    """Run the shifted-page alignment for a single work and return a per-page
    frame for review/visualization. Intended for notebook use.

    Builds on ``align_shifted_pages(detailed=True)`` and adds derived review
    columns: ``is_anchor`` / ``is_matched`` (flags inferred from ``shift`` /
    ``page_filename``), ``text_len`` / ``zip_text_len`` (character counts),
    ``text_snippet`` / ``zip_text_snippet`` (first-line hover snippets), and
    ``match_score`` (the rapidfuzz.fuzz.ratio, 0-100, between each matched page
    and its *actual* aligned zip page) so it can be compared against
    ``cdist_best_score`` (the best score against *any* zip page).

    ``pages`` may be a page DataFrame (with ``id``, ``order``/``text`` columns) or
    a list of page dicts; ``zipfile`` may be an open ``ZipFile`` or a path to one.
    """
    pages_df = pages if isinstance(pages, pl.DataFrame) else pl.DataFrame(pages)
    # add an order column (numeric trailing page id) if the caller didn't supply one
    if "order" not in pages_df.columns:
        pages_df = pages_df.with_columns(
            order=pl.col.id.str.extract(r"([0-9]+$)").cast(pl.Int64)
        )

    # accept an open ZipFile or a path; open+close our own handle for a path
    if isinstance(zipfile, ZipFile):
        zip_pages_df = get_zipfile_pages(zipfile)
    else:
        with ZipFile(zipfile) as zf:
            zip_pages_df = get_zipfile_pages(zf)

    logger.info("reviewing alignment for %s (%d pages)", work_id, pages_df.height)
    detailed_df = align_shifted_pages(pages_df, zip_pages_df, detailed=True)

    # add derived review fields inferred from the raw alignment columns
    return detailed_df.with_columns(
        # a page is an anchor if it contributed a trusted (non-null) shift;
        is_anchor=pl.col.shift.is_not_null(),
        # is_matched reflects the final (post-dedup) filename assignment
        is_matched=pl.col.page_filename.is_not_null(),
        # character counts (zip_text is null for unmatched pages)
        text_len=pl.col.text.str.len_chars(),
        zip_text_len=pl.col.zip_text.str.len_chars(),
        # first-line snippets for hover text
        text_snippet=_text_snippet_expr("text"),
        zip_text_snippet=_text_snippet_expr("zip_text"),
        # similarity of each matched page against its *actual* aligned zip page
        # (0-100), to compare against cdist_best_score (best against any zip page)
        match_score=pl.when(pl.col.page_filename.is_not_null())
        .then(pds.str_fuzz("text", "zip_text", parallel=False) * 100)
        .otherwise(None),
    )


def plot_alignment(review_df: pl.DataFrame):
    """Visualize a review frame from ``review_alignment`` as an Altair chart.

    Original pages are drawn on one row and their mapped zip pages on a second
    row, both positioned by page order along x. A line connects each original
    page to the zip page it maps to, so the horizontal offset of that line is the
    shift; parallel lines mean a consistent shift, and crossings/gaps stand out.
    Points are colored by alignment status (anchor / inferred / unmatched) and
    sized by match score. Hover shows a compact mapping/score line (e.g.
    "page 24 -> zip 12 | best 100 | aligned 100") and a text snippet.

    Expects the derived columns added by ``review_alignment`` (snippets,
    ``match_score``). Requires ``altair`` (install the ``notebooks`` extra).
    """
    try:
        import altair as alt
    except ImportError as err:  # pragma: no cover - notebook-only dependency
        raise ImportError(
            "plot_alignment requires altair; install the 'notebooks' extra "
            "(pip install 'corppa[notebooks]')"
        ) from err

    # readable status for color/legend, plus a single "info" line summarizing the
    # order mapping (orig -> zip) and scores, so the hover card stays compact
    prepared = review_df.with_columns(
        status=pl.when(~pl.col.is_matched)
        .then(pl.lit("unmatched"))
        .when(pl.col.is_anchor)
        .then(pl.lit("anchor"))
        .otherwise(pl.lit("inferred")),
        info=(
            pl.format(
                "page {} → zip {}",
                pl.col.order,
                pl.when(pl.col.is_matched)
                .then(pl.col.aligned_order.cast(pl.String))
                .otherwise(pl.lit("—")),
            )
            # append scores when available (nulls are shown as blanks)
            + pl.when(pl.col.cdist_best_score.is_not_null())
            .then(pl.format(" | best {}", pl.col.cdist_best_score.round(1)))
            .otherwise(pl.lit(""))
            + pl.when(pl.col.match_score.is_not_null())
            .then(pl.format(" | aligned {}", pl.col.match_score.round(1)))
            .otherwise(pl.lit(""))
        ),
    )

    # reshape to long form: two points per page, one on the "original" row (x =
    # original order) and one on the "mapped zip" row (x = aligned order). Sharing
    # the page id lets a line connect them; the horizontal gap is the shift.
    orig_points = prepared.select(
        pl.col.id,
        row=pl.lit("original"),
        x=pl.col.order,
        status=pl.col.status,
        # drives point size; coalesce null (unmatched/unscored) to 0 so the point
        # still renders (a null size value would drop the mark entirely)
        match_score=pl.col.match_score.fill_null(0.0),
        # whether the page has a positive aligned-match score (filled vs hollow)
        has_match=pl.col.match_score.fill_null(0.0) > 0,
        info=pl.col.info,
        # snippet reflects this row's own text (original page)
        snippet=pl.col.text_snippet,
    )
    # only matched pages get a mapped-zip point / connecting line
    zip_points = prepared.filter(pl.col.is_matched).select(
        pl.col.id,
        row=pl.lit("mapped zip"),
        x=pl.col.aligned_order,
        status=pl.col.status,
        match_score=pl.col.match_score.fill_null(0.0),
        has_match=pl.col.match_score.fill_null(0.0) > 0,
        info=pl.col.info,
        # snippet reflects this row's own text (matched zip page)
        snippet=pl.col.zip_text_snippet,
    )
    plot_df = pl.concat([orig_points, zip_points])

    # PPA palette: anchor = french-blue (trusted), inferred = seafoam-blue
    # (softer/derived), unmatched = rosy-pink (needs attention)
    status_scale = alt.Scale(
        domain=["anchor", "inferred", "unmatched"],
        range=["#4661ac", "#57c4c4", "#f05b69"],
    )
    status_color = alt.Color(
        "status:N",
        scale=status_scale,
        title="alignment",
        # pin the legend swatch size/opacity so it stays visible independent of
        # the size (match score) encoding, which otherwise shrinks the swatches
        legend=alt.Legend(symbolType="circle", symbolSize=120, symbolOpacity=1.0),
    )
    # keep the two rows in a fixed top/bottom order
    row_scale = alt.Scale(domain=["original", "mapped zip"])

    # constrain the x-axis to the pages actually plotted. For excerpts the zip
    # holds the whole volume while only a few excerpt pages are present, so an
    # auto-domain would stretch across the full volume and squash the points;
    # derive the domain from the plotted orders (both rows) with a little padding.
    x_vals = plot_df["x"].drop_nulls()
    x_min, x_max = x_vals.min(), x_vals.max()
    pad = max(1, round((x_max - x_min) * 0.02))
    x_scale = alt.Scale(domain=[x_min - pad, x_max + pad], nice=False)

    tooltip = [
        alt.Tooltip("info:N", title="mapping"),
        alt.Tooltip("snippet:N", title="text"),
    ]

    # altair 5+ accepts a polars DataFrame directly via the dataframe interchange
    # protocol, so no pandas conversion is needed
    base = alt.Chart(plot_df)

    # connecting line between each page's two points (drawn only for matched pages,
    # which are the only ones with both endpoints). Uses the same color encoding
    # as the points so the shared color legend merges cleanly (a per-layer
    # legend=None here would suppress the merged legend entirely).
    lines = base.mark_line(opacity=0.4).encode(
        x=alt.X("x:Q", title="page order", scale=x_scale),
        y=alt.Y("row:N", scale=row_scale, title=None),
        detail="id:N",
        color=status_color,
    )
    # circular points sized by aligned-match score. filled=True so the single
    # color encoding fills the dot and drives a proper (filled-swatch) legend;
    # fillOpacity is toggled so scored pages read as solid dots and zero-/no-score
    # pages (e.g. unmatched original pages) read as hollow rings instead of
    # disappearing. The status color is also applied to the stroke so hollow
    # points still have a visible (colored) outline when the fill is transparent.
    points = base.mark_point(
        shape="circle", strokeWidth=1.5, filled=True, strokeOpacity=1.0
    ).encode(
        x=alt.X("x:Q", title="page order", scale=x_scale),
        y=alt.Y("row:N", scale=row_scale, title=None),
        color=status_color,
        stroke=status_color,
        # solid fill when there is a positive match score, hollow (transparent) at 0
        fillOpacity=alt.condition("datum.has_match", alt.value(1.0), alt.value(0.0)),
        size=alt.Size(
            "match_score:Q",
            title="match score",
            # floor the size range so a zero score still has a visible ring
            scale=alt.Scale(range=[60, 300]),
        ),
        tooltip=tooltip,
    )
    return (
        (lines + points)
        .resolve_scale(size="independent")
        .properties(
            width=700,
            height=220,
            title="page alignment (original vs mapped zip order)",
        )
        .interactive()
    )


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


def get_ht_zipfile_path(work_id: str, image_dir: Path) -> Path:
    """Return the expected HathiTrust zip path for a work under ``image_dir``.
    The zip is named from the (encoded) HathiTrust id without institution prefix."""
    htid = get_volume_id(work_id)
    # must be encoded to convert ark style ids to a file-safe format
    encoded = encode_htid(htid)
    htid_suffix = encoded.split(".")[-1]
    return image_dir / "HathiTrust" / encoded / f"{htid_suffix}.zip"


def process_ht_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    htid = get_volume_id(work_id)
    htid_suffix = encode_htid(htid).split(".")[-1]
    zipfile_path = get_ht_zipfile_path(work_id, image_dir)
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
