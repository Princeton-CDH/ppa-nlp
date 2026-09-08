# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import NonCallableMock, call, patch

import pytest

from corppa.poetry_detection.core import Span
from corppa.poetry_detection.evaluation.evaluate_poetry_spans import (
    PageEvaluation,
    PageSpans,
    get_page_eval,
    get_page_evals,
    write_page_evals,
)


class TestPageSpans:
    @patch.object(PageSpans, "_get_unlabeled_spans", autospec=True)
    @patch.object(PageSpans, "_get_labeled_spans", autospec=True)
    def test_init(self, mock_get_labeled_spans, mock_get_unlabeled_spans):
        mock_get_labeled_spans.return_value = "labeled spans list"
        mock_get_unlabeled_spans.return_value = "unlabeled spans list"
        page_json = {"page_id": "12345"}
        # Defaults
        result = PageSpans(page_json)
        assert result.page_id == "12345"
        assert result.labeled_spans == "labeled spans list"
        assert result.unlabeled_spans == "unlabeled spans list"
        mock_get_labeled_spans.assert_called_once_with(page_json, index_pfx="")
        mock_get_unlabeled_spans.assert_called_once_with("labeled spans list")
        # With optional arg
        mock_get_labeled_spans.reset_mock()
        mock_get_unlabeled_spans.reset_mock()
        result = PageSpans(page_json, index_pfx="pfx")
        mock_get_labeled_spans.assert_called_once_with(page_json, index_pfx="pfx")
        mock_get_unlabeled_spans.assert_called_once_with("labeled spans list")

    def test_get_labeled_spans(self):
        # No spans
        page_json = {"page_id": "id", "poem_spans": []}
        result = PageSpans._get_labeled_spans(page_json)
        assert result == []

        # With labeled spans
        poem_spans = [
            {"start": 0, "end": 5, "poem_id": "c"},
            {"start": 3, "end": 4, "poem_id": "a"},
            {"start": 1, "end": 8, "poem_id": "b"},
            {"start": 1, "end": 3, "poem_id": "c"},
        ]
        page_json = {"page_id": "id", "poem_spans": poem_spans}
        result = PageSpans._get_labeled_spans(page_json)
        expected_spans = [
            Span(0, 5, "c"),
            Span(1, 3, "c"),
            Span(1, 8, "b"),
            Span(3, 4, "a"),
        ]
        assert len(result) == len(poem_spans)
        for i, span in enumerate(result):
            assert span == expected_spans[i]

        # With unlabeled spans
        unlabeled_spans = [
            {"start": 0, "end": 5, "other": "ignored"},
            {"start": 10, "end": 20},
        ]
        page_json = {"page_id": "id", "poem_spans": unlabeled_spans}
        result = PageSpans._get_labeled_spans(page_json)
        expected_spans = [Span(0, 5, ""), Span(10, 20, "")]
        assert len(result) == len(unlabeled_spans)
        for i, span in enumerate(result):
            assert span == expected_spans[i]

        # With index prefix
        ## Test unprefixed fields are ignored
        prefixed_spans = [
            {"start": 0, "end": 5, "page_start": 1, "page_end": 4, "poem_id": "a"},
            {"start": 1, "end": 8, "page_start": 2, "page_end": 6, "poem_id": "b"},
        ]
        page_json = {"page_id": "id", "poem_spans": prefixed_spans}
        result = PageSpans._get_labeled_spans(page_json, index_pfx="page_")
        expected_spans = [Span(1, 4, "a"), Span(2, 6, "b")]
        assert len(result) == len(prefixed_spans)
        for i, span in enumerate(result):
            assert span == expected_spans[i]
        ## Missing prefixed start field
        missing_start = [{"start": 0, "end": 5, "page_end": 4, "poem_id": "a"}]
        page_json = {"page_id": "id", "poem_spans": missing_start}
        with pytest.raises(
            ValueError, match="Missing span field: 'page_start'. Check index prefix."
        ):
            result = PageSpans._get_labeled_spans(page_json, index_pfx="page_")
        ## Missing prefixed end field
        missing_end = [{"start": 0, "end": 5, "page_start": 1, "poem_id": "a"}]
        page_json = {"page_id": "id", "poem_spans": missing_end}
        with pytest.raises(
            ValueError, match="Missing span field: 'page_end'. Check index prefix."
        ):
            result = PageSpans._get_labeled_spans(page_json, index_pfx="page_")

    def test_get_unlabeled_spans(self):
        # No spans
        result = PageSpans._get_unlabeled_spans([])
        assert result == []

        # With labeled spans
        labeled_spans = [
            Span(0, 1, "a"),
            Span(1, 4, "a"),
            Span(2, 3, "b"),
            Span(3, 5, "d"),
            Span(9, 10, "a"),
        ]
        result = PageSpans._get_unlabeled_spans(labeled_spans)
        expected_spans = [Span(0, 1, ""), Span(1, 5, ""), Span(9, 10, "")]
        assert len(result) == 3
        for i, span in enumerate(result):
            assert span == expected_spans[i]

        # With label-less spans
        no_labels = [Span(s.start, s.end, "") for s in labeled_spans]
        result = PageSpans._get_unlabeled_spans(no_labels)
        assert len(result) == 3
        for i, span in enumerate(result):
            assert span == expected_spans[i]


class TestPageEvaluation:
    @patch.object(PageEvaluation, "_get_span_pairs", autospec=True)
    @patch.object(PageEvaluation, "_get_span_mappings", autospec=True)
    def test_init(self, mock_get_span_mappings, mock_get_span_pairs):
        # Page id mismatch
        ref_page = NonCallableMock(spec=PageSpans, page_id="a")
        sys_page = NonCallableMock(spec=PageSpans, page_id="b")
        with pytest.raises(
            ValueError,
            match="Reference and system spans must correspond to the same page",
        ):
            PageEvaluation(ref_page, sys_page)
        mock_get_span_mappings.assert_not_called()
        mock_get_span_pairs.assert_not_called()

        # Setup for non-error cases
        ref_page = NonCallableMock(
            spec=PageSpans,
            page_id="id",
            labeled_spans="ref labeled",
            unlabeled_spans="ref unlabeled",
        )
        sys_page = NonCallableMock(
            spec=PageSpans,
            page_id="id",
            labeled_spans="sys labeled",
            unlabeled_spans="sys unlabeled",
        )

        # Default case (ignore_label = False)
        mock_get_span_mappings.return_value = (
            "ref span --> sys span",
            "sys span --> ref spans",
        )
        mock_get_span_pairs.return_value = "span pairs"
        result = PageEvaluation(ref_page, sys_page)
        assert result.page_id == "id"
        assert not result.ignore_label
        assert result.ref_spans == "ref labeled"
        assert result.sys_spans == "sys labeled"
        assert result.ref_to_sys == "ref span --> sys span"
        assert result.sys_to_refs == "sys span --> ref spans"
        assert result.span_pairs == "span pairs"
        mock_get_span_mappings.assert_called_once()
        mock_get_span_pairs.assert_called_once()

        # Ignore labels
        mock_get_span_mappings.reset_mock()
        mock_get_span_pairs.reset_mock()
        result = PageEvaluation(ref_page, sys_page, ignore_label=True)
        assert result.page_id == "id"
        assert result.ignore_label
        assert result.ref_spans == "ref unlabeled"
        assert result.sys_spans == "sys unlabeled"
        assert result.ref_to_sys == "ref span --> sys span"
        assert result.sys_to_refs == "sys span --> ref spans"
        assert result.span_pairs == "span pairs"
        mock_get_span_mappings.assert_called_once()
        mock_get_span_pairs.assert_called_once()

    def test_get_span_mappings(self):
        # Simple 1-1 cases
        ref_spans = [Span(2, 5, "a"), Span(10, 15, "b")]
        sys_spans = [Span(1, 6, "a"), Span(11, 13, "c")]
        ## Label sensitive
        results = PageEvaluation._get_span_mappings(ref_spans, sys_spans, False)
        assert results[0] == [0, None]
        assert results[1] == [[0], []]
        ## Label insensitive
        results = PageEvaluation._get_span_mappings(ref_spans, sys_spans, True)
        assert results[0] == [0, 1]
        assert results[1] == [[0], [1]]

        # Best overlap
        ref_spans = [Span(3, 8, "a")]
        sys_spans = [Span(1, 4, "a"), Span(4, 7, "a"), Span(7, 10, "a")]
        for ignore_label in [False, True]:
            results = PageEvaluation._get_span_mappings(
                ref_spans, sys_spans, ignore_label
            )
            assert results[0] == [1]
            assert results[1] == [[], [0], []]

        # System span mapping to multiple reference spans
        ref_spans = [Span(2, 5, "a"), Span(7, 11, "b"), Span(18, 20, "a")]
        sys_spans = [Span(0, 25, "a")]
        ## Label sensitive
        results = PageEvaluation._get_span_mappings(ref_spans, sys_spans, False)
        assert results[0] == [0, None, 0]
        assert results[1] == [[0, 2]]
        ## Label insensitive
        results = PageEvaluation._get_span_mappings(ref_spans, sys_spans, True)
        assert results[0] == [0, 0, 0]
        assert results[1] == [[0, 1, 2]]

    def test_get_span_pairs(self):
        # Simple 1-1 case
        ref_spans = ["a", "b", "c"]
        sys_spans = ["A", "_", "C", "_"]
        sys_to_refs = [[0], [], [2], []]
        result = PageEvaluation._get_span_pairs(ref_spans, sys_spans, sys_to_refs)
        assert result == [("a", "A"), ("c", "C")]

        # Requires system span splitting
        ## System span maps to two reference spans
        ref_spans = [Span(0, 3, "a"), Span(10, 12, "b"), Span(15, 17, "b")]
        sys_spans = [Span(0, 3, "A"), Span(8, 25, "B")]
        sys_to_refs = [[0], [1, 2]]
        expected_result = [
            (ref_spans[0], Span(0, 3, "A")),
            (ref_spans[1], Span(8, 15, "B")),
            (ref_spans[2], Span(15, 25, "B")),
        ]
        result = PageEvaluation._get_span_pairs(ref_spans, sys_spans, sys_to_refs)
        assert result == expected_result

    @patch.object(Span, "overlap_factor", autospec=True)
    @patch.object(Span, "is_exact_match", autospec=True)
    def test_relevance_score(self, mock_is_exact_match, mock_overlap_factor):
        # test pairs (note the actual values are not used directly in test)
        span_pairs = [
            (Span(0, 1, "a"), Span(0, 1, "A")),
            (Span(1, 2, "b"), Span(1, 2, "B")),
            (Span(2, 3, "c"), Span(2, 3, "C")),
        ]
        # Only exact matches
        mock_is_exact_match.return_value = True
        result = PageEvaluation._relevance_score(
            span_pairs, "ignore_label", "partial_weight"
        )
        assert result == 3
        assert mock_is_exact_match.call_count == 3
        mock_overlap_factor.assert_not_called()

        # With partial matches
        ## No downweighting
        mock_is_exact_match.reset_mock()
        mock_is_exact_match.side_effect = [True, False, False]
        mock_overlap_factor.reset_mock()
        mock_overlap_factor.side_effect = [0.2, 0.3]
        result = PageEvaluation._relevance_score(span_pairs, "ignore_label", 1)
        assert result == 1.5
        assert mock_is_exact_match.call_count == 3
        assert mock_overlap_factor.call_count == 2
        ## With downweighting
        mock_is_exact_match.reset_mock()
        mock_is_exact_match.side_effect = [True, False, False]
        mock_overlap_factor.reset_mock()
        mock_overlap_factor.side_effect = [0.2, 0.3]
        result = PageEvaluation._relevance_score(span_pairs, "ignore_label", 0.5)
        assert result == 1.25

    def test_retrieved_count(self):
        # Simple matches
        assert PageEvaluation._retrieved_count([[]]) == 1
        assert PageEvaluation._retrieved_count([[1]]) == 1
        assert PageEvaluation._retrieved_count([[3], []]) == 2
        assert PageEvaluation._retrieved_count([[], [1], [5]]) == 3

        # System span covers matches multiple reference spans
        assert PageEvaluation._retrieved_count([[3, 5]]) == 2
        assert PageEvaluation._retrieved_count([[], [1, 2, 3], [6]]) == 5

    @patch.object(PageEvaluation, "_relevance_score", autospec=True)
    @patch.object(PageEvaluation, "_retrieved_count", autospec=True)
    @patch.object(
        PageEvaluation, "_get_span_pairs", return_value="span_pairs", autospec=True
    )
    @patch.object(
        PageEvaluation,
        "_get_span_mappings",
        return_value=("r2s", "s2rs"),
        autospec=True,
    )
    def test_precision(
        self, mock_span_maps, mock_span_pairs, mock_retrieved_count, mock_relevance
    ):
        ref_page_none = NonCallableMock(spec=PageSpans, page_id="id", labeled_spans=[])
        sys_page_none = NonCallableMock(spec=PageSpans, page_id="id", labeled_spans=[])
        ref_page = NonCallableMock(spec=PageSpans, page_id="id", labeled_spans=["r"])
        sys_page = NonCallableMock(spec=PageSpans, page_id="id", labeled_spans=["s"])

        # Edge case: No system spans
        ## With reference spans
        page_eval = PageEvaluation(ref_page, sys_page_none)
        assert page_eval.precision() == 1
        mock_retrieved_count.assert_not_called()
        mock_relevance.assert_not_called()
        ## No reference spans
        page_eval = PageEvaluation(ref_page_none, sys_page_none)
        assert page_eval.precision() == 1
        mock_retrieved_count.assert_not_called()
        mock_relevance.assert_not_called()

        # With system spans
        ## Single span
        mock_retrieved_count.return_value = 1
        mock_relevance.return_value = 0.5
        page_eval = PageEvaluation(ref_page, sys_page, ignore_label="ignore_label")
        assert page_eval.precision(partial_match_weight="weight") == 0.5
        mock_retrieved_count.assert_called_once_with("s2rs")
        mock_relevance.assert_called_once_with("span_pairs", "ignore_label", "weight")
        ## Multiple spans
        mock_retrieved_count.reset_mock()
        mock_relevance.reset_mock()
        mock_retrieved_count.return_value = 2
        mock_relevance.return_value = 0.5
        sys_page_mult = NonCallableMock(
            spec=PageSpans, page_id="id", labeled_spans=["s"] * 2
        )
        page_eval = PageEvaluation(ref_page, sys_page_mult, ignore_label="ignore_label")
        assert page_eval.precision(partial_match_weight="weight") == 0.5 / 2
        mock_retrieved_count.assert_called_once_with("s2rs")
        mock_relevance.assert_called_once_with("span_pairs", "ignore_label", "weight")
        ## System spans are split
        mock_retrieved_count.reset_mock()
        mock_relevance.reset_mock()
        mock_retrieved_count.return_value = 7
        mock_relevance.return_value = 4.5
        page_eval = PageEvaluation(ref_page, sys_page_mult, ignore_label="ignore_label")
        assert page_eval.precision(partial_match_weight="weight") == 4.5 / 7
        mock_retrieved_count.assert_called_once_with("s2rs")
        mock_relevance.assert_called_once_with("span_pairs", "ignore_label", "weight")
        ## Same result with no reference spans
        page_eval = PageEvaluation(
            ref_page_none, sys_page_mult, ignore_label="ignore_label"
        )
        assert page_eval.precision(partial_match_weight="weight") == 4.5 / 7

    @patch.object(PageEvaluation, "_relevance_score", autospec=True)
    @patch.object(
        PageEvaluation, "_get_span_pairs", return_value="span_pairs", autospec=True
    )
    @patch.object(
        PageEvaluation,
        "_get_span_mappings",
        return_value=("r2s", "s2rs"),
        autospec=True,
    )
    def test_recall(self, mock_span_maps, mock_span_pairs, mock_relevance):
        ref_page_none = NonCallableMock(spec=PageSpans, page_id="id", labeled_spans=[])
        sys_page_none = NonCallableMock(spec=PageSpans, page_id="id", labeled_spans=[])
        ref_page = NonCallableMock(spec=PageSpans, page_id="id", labeled_spans=["r"])
        sys_page = NonCallableMock(spec=PageSpans, page_id="id", labeled_spans=["s"])

        # Edge case: No reference spans
        ## With system spans
        page_eval = PageEvaluation(ref_page_none, sys_page)
        assert page_eval.recall() == 1
        mock_relevance.assert_not_called()
        ## No system spans
        page_eval = PageEvaluation(ref_page_none, sys_page_none)
        assert page_eval.recall() == 1
        mock_relevance.assert_not_called()

        # Standard Case
        ## Single reference span
        mock_relevance.return_value = 0.5
        page_eval = PageEvaluation(ref_page, sys_page)
        assert page_eval.recall(partial_match_weight="weight") == 0.5 / 1
        mock_relevance.assert_called_once_with("span_pairs", False, "weight")
        ## Multiple reference spans
        mock_relevance.reset_mock()
        mock_relevance.return_value = 2.1
        ref_page_mult = NonCallableMock(
            spec=PageSpans, page_id="id", labeled_spans=["r"] * 3
        )
        page_eval = PageEvaluation(ref_page_mult, sys_page)
        assert page_eval.recall(partial_match_weight="weight") == 2.1 / 3
        mock_relevance.assert_called_once_with("span_pairs", False, "weight")
        ## Same result with no system spans
        mock_relevance.reset_mock()
        page_eval = PageEvaluation(ref_page_mult, sys_page_none)
        assert page_eval.recall() == 2.1 / 3

    @patch.object(PageEvaluation, "_relevance_score", autospec=True)
    @patch.object(
        PageEvaluation, "_get_span_pairs", return_value="span_pairs", autospec=True
    )
    @patch.object(
        PageEvaluation,
        "_get_span_mappings",
        return_value=("r2s", "s2rs"),
        autospec=True,
    )
    def test_recall_ignore_label(self, mock_span_maps, mock_span_pairs, mock_relevance):
        ref_page = NonCallableMock(spec=PageSpans, page_id="id", unlabeled_spans=["r"])
        sys_page = NonCallableMock(spec=PageSpans, page_id="id", unlabeled_spans=["s"])

        ## Single reference span
        mock_relevance.return_value = 0.5
        page_eval = PageEvaluation(ref_page, sys_page, ignore_label=True)
        assert page_eval.recall(partial_match_weight="weight") == 0.5 / 1
        mock_relevance.assert_called_once_with("span_pairs", True, "weight")

        ## Multiple reference spans
        mock_relevance.reset_mock()
        mock_relevance.return_value = 2.1
        ref_page_mult = NonCallableMock(
            spec=PageSpans, page_id="id", unlabeled_spans=["r"] * 3
        )
        page_eval = PageEvaluation(ref_page_mult, sys_page, ignore_label=True)
        assert page_eval.recall(partial_match_weight="weight") == 2.1 / 3
        mock_relevance.assert_called_once_with("span_pairs", True, "weight")

        ## Same results with no system spans
        mock_relevance.reset_mock()
        sys_page_none = NonCallableMock(
            spec=PageSpans, page_id="id", unlabeled_spans=[]
        )
        page_eval = PageEvaluation(ref_page_mult, sys_page_none, ignore_label=True)
        assert page_eval.recall() == 2.1 / 3

    @patch.object(PageEvaluation, "recall", autospec=True)
    @patch.object(PageEvaluation, "precision", autospec=True)
    @patch.object(
        PageEvaluation, "_get_span_pairs", return_value="span_pairs", autospec=True
    )
    @patch.object(
        PageEvaluation,
        "_get_span_mappings",
        return_value=("r2s", "s2rs"),
        autospec=True,
    )
    def test_f1_score(
        self, mock_span_maps, mock_span_pairs, mock_precision, mock_recall
    ):
        ref_page_none = NonCallableMock(
            spec=PageSpans, page_id="id", unlabeled_spans=[]
        )
        sys_page_none = NonCallableMock(
            spec=PageSpans, page_id="id", unlabeled_spans=[]
        )
        ref_page = NonCallableMock(spec=PageSpans, page_id="id", unlabeled_spans=["r"])
        sys_page = NonCallableMock(spec=PageSpans, page_id="id", unlabeled_spans=["s"])

        # Edge case: precision and recall are 0
        ## No system or ref spans
        mock_precision.return_value = 0
        mock_recall.return_value = 0
        page_eval = PageEvaluation(ref_page_none, sys_page_none, ignore_label=True)
        assert page_eval.f1_score() == 1
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)
        ## With system spans
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        page_eval = PageEvaluation(ref_page_none, sys_page, ignore_label=True)
        assert page_eval.f1_score() == 0
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)
        ## With ref spans
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        page_eval = PageEvaluation(ref_page, sys_page_none, ignore_label=True)
        assert page_eval.f1_score() == 0
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)
        ## With both
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        page_eval = PageEvaluation(ref_page, sys_page)
        assert page_eval.f1_score() == 0
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)

        # Standard
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        mock_precision.return_value = 0.25
        mock_recall.return_value = 0.75
        page_eval = PageEvaluation(ref_page, sys_page)
        assert page_eval.f1_score() == 0.375  # (2*.25*.75)/1
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)

        mock_precision.return_value = 0.75
        mock_recall.return_value = 0.5
        page_eval = PageEvaluation(ref_page, sys_page)
        assert page_eval.f1_score() == 0.6  # (2*.5*.75)/1.25

        ## Check propagation of optional var
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        page_eval.f1_score(partial_match_weight=0.2)
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=0.2)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=0.2)

    @patch.object(PageEvaluation, "recall", autospec=True)
    @patch.object(PageEvaluation, "precision", autospec=True)
    @patch.object(
        PageEvaluation, "_get_span_pairs", return_value="span_pairs", autospec=True
    )
    @patch.object(
        PageEvaluation,
        "_get_span_mappings",
        return_value=("r2s", "s2rs"),
        autospec=True,
    )
    def test_f1_score_ignore_label(
        self, mock_span_maps, mock_span_pairs, mock_precision, mock_recall
    ):
        ref_page_none = NonCallableMock(
            spec=PageSpans, page_id="id", unlabeled_spans=[]
        )
        sys_page_none = NonCallableMock(
            spec=PageSpans, page_id="id", unlabeled_spans=[]
        )
        ref_page = NonCallableMock(spec=PageSpans, page_id="id", unlabeled_spans=["r"])
        sys_page = NonCallableMock(spec=PageSpans, page_id="id", unlabeled_spans=["s"])

        # Edge case: precision and recall are 0
        ## No system or ref spans
        mock_precision.return_value = 0
        mock_recall.return_value = 0
        page_eval = PageEvaluation(ref_page_none, sys_page_none, ignore_label=True)
        assert page_eval.f1_score() == 1
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)
        ## With system spans
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        page_eval = PageEvaluation(ref_page_none, sys_page, ignore_label=True)
        assert page_eval.f1_score() == 0
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)
        ## With ref spans
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        page_eval = PageEvaluation(ref_page, sys_page_none, ignore_label=True)
        assert page_eval.f1_score() == 0
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)
        ## With both
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        page_eval = PageEvaluation(ref_page, sys_page, ignore_label=True)
        assert page_eval.f1_score() == 0
        mock_precision.assert_called_once_with(page_eval, partial_match_weight=1)
        mock_recall.assert_called_once_with(page_eval, partial_match_weight=1)

    def test_get_match_counts(self):
        expected_results = {
            "n_span_matches": 0,
            "n_span_misses": 0,
            "n_poem_matches": 0,
            "n_poem_misses": 0,
        }
        # Empty case
        results = PageEvaluation._get_match_counts([], [])
        assert results == expected_results

        # Typical cases
        spans = [Span(0, 1, "a"), Span(0, 1, "a"), Span(0, 1, "b"), Span(0, 1, "c")]
        expected_results["n_span_matches"] = 3
        expected_results["n_span_misses"] = 1
        expected_results["n_poem_matches"] = 2
        expected_results["n_poem_misses"] = 1
        results = PageEvaluation._get_match_counts(spans, [1, 3, None, 2])
        assert results == expected_results

        results = PageEvaluation._get_match_counts(spans, [None, None, None, 0])
        expected_results["n_span_matches"] = 1
        expected_results["n_span_misses"] = 3
        expected_results["n_poem_matches"] = 1
        expected_results["n_poem_misses"] = 2
        assert results == expected_results

    def test_get_spurious_counts(self):
        expected_results = {
            "n_span_spurious": 0,
            "n_poem_spurious": 0,
        }
        # Empty case
        results = PageEvaluation._get_spurious_counts([], [], [])
        assert results == expected_results

        # Simple no spurious
        ref_spans = [Span(0, 1, "a"), Span(0, 1, "b"), Span(0, 1, "c")]
        sys_spans = [Span(0, 1, "a"), Span(0, 1, "b"), Span(0, 1, "c")]
        sys_to_refs = [[0], [1], [2]]
        results = PageEvaluation._get_spurious_counts(ref_spans, sys_spans, sys_to_refs)
        assert results == expected_results

        # Single Spurious span
        ## ...but not poem
        sys_spans = [Span(0, 1, "a"), Span(0, 1, "a"), Span(0, 1, "c")]
        sys_to_refs = [[0], [], [2]]
        expected_results["n_span_spurious"] = 1
        results = PageEvaluation._get_spurious_counts(ref_spans, sys_spans, sys_to_refs)
        assert results == expected_results
        # ...and poem
        sys_spans = [Span(0, 1, "a"), Span(0, 1, "d"), Span(0, 1, "c")]
        expected_results["n_poem_spurious"] = 1
        results = PageEvaluation._get_spurious_counts(ref_spans, sys_spans, sys_to_refs)
        assert results == expected_results

        # More complex example
        sys_spans = [
            Span(0, 1, "a"),
            Span(0, 1, "d"),
            Span(0, 1, "c"),
            Span(0, 1, "d"),
            Span(0, 1, "b"),
        ]
        sys_to_refs = [[0, 1], [], [2], [], []]
        results = PageEvaluation._get_spurious_counts(ref_spans, sys_spans, sys_to_refs)
        expected_results["n_span_spurious"] = 3
        expected_results["n_poem_spurious"] = 1
        assert results == expected_results

    @patch.object(PageEvaluation, "f1_score", return_value="f1_score", autospec=True)
    @patch.object(PageEvaluation, "recall", return_value="recall_score", autospec=True)
    @patch.object(
        PageEvaluation, "precision", return_value="precision_score", autospec=True
    )
    @patch.object(PageEvaluation, "_get_spurious_counts", autospec=True)
    @patch.object(PageEvaluation, "_get_match_counts", autospec=True)
    @patch.object(PageEvaluation, "_get_span_pairs", autospec=True)
    @patch.object(
        PageEvaluation,
        "_get_span_mappings",
        return_value=("r2s", "s2rs"),
        autospec=True,
    )
    def test_evaluate(
        self,
        mock_span_maps,
        mock_span_pairs,
        mock_matches,
        mock_spurious,
        mock_precision,
        mock_recall,
        mock_f1,
    ):
        # Setup PageSpans inputs
        ref_page = NonCallableMock(
            spec=PageSpans,
            page_id="id",
            labeled_spans="ref labeled",
            unlabeled_spans="ref unlabeled",
        )
        sys_page = NonCallableMock(
            spec=PageSpans,
            page_id="id",
            labeled_spans="sys labeled",
            unlabeled_spans="sys unlabeled",
        )

        # Set mock count values
        mock_matches.return_value = {
            "n_span_matches": "a",
            "n_span_misses": "b",
            "n_poem_matches": "c",
            "n_poem_misses": "d",
        }
        mock_spurious.return_value = {
            "n_span_spurious": "A",
            "n_poem_spurious": "B",
        }
        # Expected result
        expected_result = {
            "page_id": "id",
            "precision": "precision_score",
            "recall": "recall_score",
            "f1": "f1_score",
            "n_span_matches": "a",
            "n_span_misses": "b",
            "n_span_spurious": "A",
            "n_poem_matches": "c",
            "n_poem_misses": "d",
            "n_poem_spurious": "B",
        }

        ## Evaluate with labels
        page_eval = PageEvaluation(ref_page, sys_page)
        result = page_eval.evaluate(partial_match_weight="partial_match_weight")
        assert result == expected_result
        mock_matches.assert_called_once_with("ref labeled", "r2s")
        mock_spurious.assert_called_once_with("ref labeled", "sys labeled", "s2rs")
        mock_precision.assert_called_once_with(
            page_eval, partial_match_weight="partial_match_weight"
        )
        mock_recall.assert_called_once_with(
            page_eval, partial_match_weight="partial_match_weight"
        )
        mock_f1.assert_called_once_with(
            page_eval, partial_match_weight="partial_match_weight"
        )

        ## Evaluate ignoring labels
        mock_matches.reset_mock()
        mock_spurious.reset_mock()
        mock_precision.reset_mock()
        mock_recall.reset_mock()
        mock_f1.reset_mock()

        page_eval = PageEvaluation(ref_page, sys_page, ignore_label=True)
        result = page_eval.evaluate(partial_match_weight="partial_match_weight")
        assert result == expected_result
        mock_matches.assert_called_once_with("ref unlabeled", "r2s")
        mock_spurious.assert_called_once_with("ref unlabeled", "sys unlabeled", "s2rs")
        mock_precision.assert_called_once_with(
            page_eval, partial_match_weight="partial_match_weight"
        )
        mock_recall.assert_called_once_with(
            page_eval, partial_match_weight="partial_match_weight"
        )
        mock_f1.assert_called_once_with(
            page_eval, partial_match_weight="partial_match_weight"
        )


@patch(
    "corppa.poetry_detection.evaluation.evaluate_poetry_spans.PageEvaluation",
    autospec=True,
)
@patch(
    "corppa.poetry_detection.evaluation.evaluate_poetry_spans.PageSpans", autospec=True
)
def test_get_page_eval(mock_page_spans, mock_page_eval):
    mock_page_spans.side_effect = ["ref-page", "sys-page"]
    mock_page_eval.return_value = "page_eval"

    # With defaults
    result = get_page_eval("ref_json", "sys_json")
    assert result == "page_eval"
    mock_page_spans.assert_has_calls(
        [call("ref_json", index_pfx=""), call("sys_json", index_pfx="")]
    )
    mock_page_eval.assert_called_once_with("ref-page", "sys-page", ignore_label=False)
    # Optional args set
    mock_page_spans.reset_mock()
    mock_page_eval.reset_mock()
    mock_page_spans.side_effect = ["ref-page", "sys-page"]
    result = get_page_eval(
        "ref_json",
        "sys_json",
        ignore_label="ignore_label",
        ref_index_pfx="ref",
        sys_index_pfx="sys",
    )
    mock_page_spans.assert_has_calls(
        [call("ref_json", index_pfx="ref"), call("sys_json", index_pfx="sys")]
    )
    mock_page_eval.assert_called_once_with(
        "ref-page", "sys-page", ignore_label="ignore_label"
    )


@patch(
    "corppa.poetry_detection.evaluation.evaluate_poetry_spans.get_page_eval",
    autospec=True,
)
def test_get_page_evals(mock_get_page_eval, tmp_path):
    ref_jsonl = '{"page_id":"a", "excerpts":[]}\n{"page_id":"b", "excerpts":[]}\n'
    ref_file = tmp_path / "ref.jsonl"
    ref_file.write_text(ref_jsonl)
    sys_jsonl = '{"page_id":"b", "poem_spans":[]}\n{"page_id":"a", "poem_spans":[]}\n'
    sys_file = tmp_path / "sys.jsonl"
    sys_file.write_text(sys_jsonl)
    mock_get_page_eval.side_effect = ["A", "B"]

    # With defaults
    results = get_page_evals(ref_file, sys_file, disable_progress=True)
    assert list(results) == ["A", "B"]
    assert mock_get_page_eval.call_count == 2
    expected_calls = [
        call(
            {"page_id": "a", "excerpts": []},
            {"page_id": "a", "poem_spans": []},
            ignore_label=False,
            ref_index_pfx="",
            sys_index_pfx="",
        ),
        call(
            {"page_id": "b", "excerpts": []},
            {"page_id": "b", "poem_spans": []},
            ignore_label=False,
            ref_index_pfx="",
            sys_index_pfx="",
        ),
    ]
    mock_get_page_eval.assert_has_calls(expected_calls)

    # With optional args
    mock_get_page_eval.reset_mock()
    mock_get_page_eval.side_effect = ["A", "B"]
    results = get_page_evals(
        ref_file,
        sys_file,
        ignore_label="flag",
        ref_index_pfx="ref",
        sys_index_pfx="sys",
        disable_progress=True,
    )
    assert list(results) == ["A", "B"]
    assert mock_get_page_eval.call_count == 2
    expected_calls = [
        call(
            {"page_id": "a", "excerpts": []},
            {"page_id": "a", "poem_spans": []},
            ignore_label="flag",
            ref_index_pfx="ref",
            sys_index_pfx="sys",
        ),
        call(
            {"page_id": "b", "excerpts": []},
            {"page_id": "b", "poem_spans": []},
            ignore_label="flag",
            ref_index_pfx="ref",
            sys_index_pfx="sys",
        ),
    ]
    mock_get_page_eval.assert_has_calls(expected_calls)


@patch(
    "corppa.poetry_detection.evaluation.evaluate_poetry_spans.get_page_evals",
    autospec=True,
)
def test_write_page_evals(mock_get_page_evals, tmp_path):
    out_csv = tmp_path / "result.csv"
    page_eval_a = NonCallableMock(spec=PageEvaluation)
    page_eval_a.evaluate.return_value = {
        "page_id": "a",
        "precision": 1,
        "recall": 1,
        "f1": 1,
        "n_span_matches": 0,
        "n_span_misses": 0,
        "n_span_spurious": 0,
        "n_poem_matches": 0,
        "n_poem_misses": 0,
        "n_poem_spurious": 0,
    }
    page_eval_b = NonCallableMock(spec=PageEvaluation)
    page_eval_b.evaluate.return_value = {
        "page_id": "b",
        "precision": 2 / 3,
        "recall": 1 / 3,
        "f1": 2 / 9,
        "n_span_matches": 2,
        "n_span_misses": 1,
        "n_span_spurious": 3,
        "n_poem_matches": 1,
        "n_poem_misses": 0,
        "n_poem_spurious": 1,
    }

    mock_get_page_evals.return_value = [page_eval_a, page_eval_b]

    write_page_evals(
        "ref_file",
        "sys_file",
        out_csv,
        ignore_label="bool",
        ref_index_pfx="ref",
        sys_index_pfx="sys",
        partial_match_weight="weight",
        disable_progress="bool",
    )
    mock_get_page_evals.assert_called_once_with(
        "ref_file",
        "sys_file",
        ignore_label="bool",
        ref_index_pfx="ref",
        sys_index_pfx="sys",
        disable_progress="bool",
    )
    page_eval_a.evaluate.assert_called_once_with(partial_match_weight="weight")
    page_eval_b.evaluate.assert_called_once_with(partial_match_weight="weight")

    fieldnames = [
        "page_id",
        "precision",
        "recall",
        "f1",
        "n_span_matches",
        "n_span_misses",
        "n_span_spurious",
        "n_poem_matches",
        "n_poem_misses",
        "n_poem_spurious",
    ]
    # Validate output file contents
    expected_lines = [
        ",".join(fieldnames) + "\n",
        "a,1,1,1,0,0,0,0,0,0\n",
        f"b,{2/3},{1/3},{2/9},2,1,3,1,0,1\n",
    ]
    expected_text = "".join(expected_lines)
    assert out_csv.read_text() == expected_text
