import sys
from unittest.mock import NonCallableMock, call, patch

import pytest

from corppa.poetry_detection.evaluation.evaluate_poetry_spans import (
    PageEvaluation,
    PageReferenceSpans,
    PageSystemSpans,
    Span,
)


class TestSpan:
    def test_init(self):
        # Invalid range: end index < start index
        error_message = "Start index must be less than end index"
        with pytest.raises(ValueError, match=error_message):
            span = Span(9, 2, "label")
        # Invalid range: end index = < start index
        with pytest.raises(ValueError, match=error_message):
            span = Span(2, 2, "label")

        # Normal case
        span = Span(2, 5, "label")
        assert span.start == 2
        assert span.end == 5
        assert span.label == "label"

    def test_len(self):
        assert len(Span(2, 5, "label")) == 3
        assert len(Span(0, 42, "label")) == 42

    def test_eq(self):
        span_a = Span(3, 6, "label")
        assert span_a == Span(3, 6, "label")
        assert span_a != Span(4, 8, "label")
        assert span_a != Span(3, 6, "different")

    def test_repr(self):
        span_a = Span(3, 6, "label")
        assert repr(span_a) == "Span(3, 6, label)"

    def test_has_overlap(self):
        span_a = Span(3, 6, "label")

        for label in ["label", "different"]:
            ## exact overlap
            span_b = Span(3, 6, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            ## partial overlap: subsets
            span_b = Span(4, 5, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(3, 5, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(4, 6, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            ## partial overlap: not subsets
            span_b = Span(1, 5, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(4, 8, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            ## no overlap
            span_b = Span(0, 1, label)
            assert not span_a.has_overlap(span_b)
            assert not span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(0, 3, label)
            assert not span_a.has_overlap(span_b)
            assert not span_a.has_overlap(span_b, ignore_label=True)

    def test_is_exact_match(self):
        span_a = Span(3, 6, "label")

        for label in ["label", "different"]:
            # exact overlap
            span_b = Span(3, 6, label)
            assert span_a.is_exact_match(span_b) == (label == "label")
            assert span_a.is_exact_match(span_b, ignore_label=True)
            # partial overlap
            span_b = Span(3, 5, label)
            assert not span_a.is_exact_match(span_b)
            assert not span_a.is_exact_match(span_b, ignore_label=True)
            span_b = Span(2, 8, label)
            assert not span_a.is_exact_match(span_b)
            assert not span_a.is_exact_match(span_b, ignore_label=True)
            # no overlap
            span_b = Span(0, 1, label)
            assert not span_a.is_exact_match(span_b)
            assert not span_a.is_exact_match(span_b, ignore_label=True)
            span_b = Span(0, 3, label)
            assert not span_a.is_exact_match(span_b)
            assert not span_a.is_exact_match(span_b, ignore_label=True)

    @patch.object(Span, "has_overlap")
    def test_overlap_length(self, mock_has_overlap):
        span_a = Span(3, 6, "label")

        # no overlap
        mock_has_overlap.return_value = False
        assert span_a.overlap_length("other span", ignore_label="bool") == 0
        mock_has_overlap.assert_called_once_with("other span", ignore_label="bool")

        # has overlap
        mock_has_overlap.reset_mock()
        mock_has_overlap.return_value = True
        ## exact overlap
        span_b = Span(3, 6, "label")
        assert span_a.overlap_length(span_b) == 3
        mock_has_overlap.assert_called_once_with(span_b, ignore_label=False)
        ## partial overlap
        span_b = Span(3, 5, "label")
        assert span_a.overlap_length(span_b) == 2
        span_b = Span(2, 8, "label")
        assert span_a.overlap_length(span_b) == 3

    @patch.object(Span, "has_overlap")
    @patch.object(Span, "overlap_length")
    def test_overlap_factor(self, mock_overlap_length, mock_has_overlap):
        span_a = Span(3, 6, "label")

        # no overlap
        mock_has_overlap.return_value = False
        assert span_a.overlap_factor("other span", ignore_label="bool") == 0
        mock_has_overlap.assert_called_once_with("other span", ignore_label="bool")
        mock_overlap_length.assert_not_called()

        # has overlap
        mock_has_overlap.reset_mock()
        mock_has_overlap.return_value = True
        mock_overlap_length.return_value = 3
        ## exact overlap
        span_b = Span(3, 6, "label")
        assert span_a.overlap_factor(span_b, ignore_label="bool") == 1
        mock_has_overlap.assert_called_once_with(span_b, ignore_label="bool")
        mock_overlap_length.assert_called_once_with(span_b, ignore_label="bool")
        ## partial overlap
        mock_has_overlap.reset_mock()
        mock_has_overlap.return_value = True
        mock_overlap_length.reset_mock()
        mock_overlap_length.return_value = 2
        span_b = Span(3, 5, "label")
        assert span_a.overlap_factor(span_b) == 2 / 3
        mock_has_overlap.assert_called_once_with(span_b, ignore_label=False)
        mock_overlap_length.assert_called_once_with(span_b, ignore_label=False)
        span_b = Span(2, 8, "label")
        mock_overlap_length.return_value = 3
        assert span_a.overlap_factor(span_b) == 3 / 6


class TestPageReferenceSpans:
    @patch.object(PageReferenceSpans, "_get_spans")
    def test_init(self, mock_get_spans):
        mock_get_spans.return_value = "spans list"
        page_json = {"page_id": "12345"}
        result = PageReferenceSpans(page_json)
        assert result.page_id == "12345"
        assert result.spans == "spans list"
        mock_get_spans.assert_called_once_with(page_json)

    def test_get_spans(self):
        # Note: Initialization calls _get_spans

        # No spans
        page_json = {"page_id": "id", "n_excerpts": 0}
        result = PageReferenceSpans(page_json)
        assert result.spans == []

        # With spans
        excerpts = [
            {"start": 0, "end": 5, "poem_id": "c"},
            {"start": 3, "end": 4, "poem_id": "a"},
            {"start": 1, "end": 8, "poem_id": "b"},
            {"start": 1, "end": 3, "poem_id": "c"},
        ]
        page_json = {"page_id": "id", "n_excerpts": 4, "excerpts": excerpts}
        result = PageReferenceSpans(page_json)
        expected_spans = [
            Span(0, 5, "c"),
            Span(1, 3, "c"),
            Span(1, 8, "b"),
            Span(3, 4, "a"),
        ]
        assert len(result.spans) == 4
        for i, span in enumerate(result.spans):
            assert span == expected_spans[i]


class TestPageSystemSpans:
    @patch.object(PageSystemSpans, "_get_unlabeled_spans")
    @patch.object(PageSystemSpans, "_get_labeled_spans")
    def test_init(self, mock_get_labeled_spans, mock_get_unlabeled_spans):
        mock_get_labeled_spans.return_value = "labeled spans list"
        mock_get_unlabeled_spans.return_value = "unlabeled spans list"
        page_json = {"page_id": "12345"}
        result = PageSystemSpans(page_json)
        assert result.page_id == "12345"
        assert result.labeled_spans == "labeled spans list"
        assert result.unlabeled_spans == "unlabeled spans list"
        mock_get_labeled_spans.assert_called_once_with(page_json)
        mock_get_unlabeled_spans.assert_called_once()

    @patch.object(PageSystemSpans, "_get_unlabeled_spans")
    def test_get_labeled_spans(self, mock_get_unlabeled_spans):
        # Note: Initialization calls _get_labeled_spans & _get_unlabeled_spans

        # No spans
        page_json = {"page_id": "id", "n_spans": 0}
        result = PageSystemSpans(page_json)
        assert result.labeled_spans == []

        # With spans
        poem_spans = [
            {"page_start": 0, "page_end": 5, "ref_id": "c"},
            {"page_start": 3, "page_end": 4, "ref_id": "a"},
            {"page_start": 1, "page_end": 8, "ref_id": "b"},
            {"page_start": 1, "page_end": 3, "ref_id": "c"},
        ]
        page_json = {"page_id": "id", "n_spans": 4, "poem_spans": poem_spans}
        result = PageSystemSpans(page_json)
        expected_spans = [
            Span(0, 5, "c"),
            Span(1, 3, "c"),
            Span(1, 8, "b"),
            Span(3, 4, "a"),
        ]
        assert len(result.labeled_spans) == 4
        for i, span in enumerate(result.labeled_spans):
            assert span == expected_spans[i]

    @patch.object(PageSystemSpans, "_get_labeled_spans")
    def test_get_unlabeled_spans(self, mock_get_labeled_spans):
        # Note: Initialization calls _get_labeled_spans & _get_unlabeled_spans

        # No spans
        mock_get_labeled_spans.return_value = []
        result = PageSystemSpans({"page_id": "id"})
        assert result.unlabeled_spans == []

        # With spans
        mock_get_labeled_spans.return_value = [
            Span(0, 1, "a"),
            Span(1, 4, "a"),
            Span(2, 3, "b"),
            Span(3, 5, "d"),
            Span(9, 10, "a"),
        ]
        result = PageSystemSpans({"page_id": "id"})
        expected_spans = [Span(0, 1, ""), Span(1, 5, ""), Span(9, 10, "")]
        assert len(result.unlabeled_spans) == 3
        for i, span in enumerate(result.unlabeled_spans):
            assert span == expected_spans[i]


class TestPageEvaluation:
    @patch.object(PageEvaluation, "_get_span_pairs")
    @patch.object(PageEvaluation, "_get_span_mappings")
    def test_init(self, mock_get_span_mappings, mock_get_span_pairs):
        # Page id mismatch
        page_ref_spans = NonCallableMock(page_id="a")
        page_sys_spans = NonCallableMock(page_id="b")
        with pytest.raises(
            ValueError,
            match="Reference and system spans must correspond to the same page",
        ):
            PageEvaluation(page_ref_spans, page_sys_spans)
            mock_get_span_mappings.assert_not_called()
            mock_get_span_pairs.assert_not_called()

        # Setup for non-error cases
        page_ref_spans = NonCallableMock(page_id="id", spans="spans")
        page_sys_spans = NonCallableMock(
            page_id="id",
            labeled_spans="labeled spans",
            unlabeled_spans="unlabeled spans",
        )

        # Default case (ignore_label = False)
        mock_get_span_mappings.return_value = (
            "ref span --> sys span",
            "sys span --> ref spans",
        )
        mock_get_span_pairs.return_value = "span pairs"
        result = PageEvaluation(page_ref_spans, page_sys_spans)
        assert result.ignore_label == False
        assert result.ref_spans == "spans"
        assert result.sys_spans == "labeled spans"
        assert result.ref_to_sys == "ref span --> sys span"
        assert result.sys_to_refs == "sys span --> ref spans"
        assert result.span_pairs == "span pairs"
        mock_get_span_mappings.assert_called_once()
        mock_get_span_pairs.assert_called_once()

        # Ignore labels
        mock_get_span_mappings.reset_mock()
        mock_get_span_mappings.return_value = (
            "ref span --> sys span",
            "sys span --> ref spans",
        )
        mock_get_span_pairs.reset_mock()
        mock_get_span_pairs.return_value = "span pairs"
        result = PageEvaluation(page_ref_spans, page_sys_spans, ignore_label=True)
        assert result.ignore_label == True
        assert result.ref_spans == "spans"
        assert result.sys_spans == "unlabeled spans"
        assert result.ref_to_sys == "ref span --> sys span"
        assert result.sys_to_refs == "sys span --> ref spans"
        assert result.span_pairs == "span pairs"
        mock_get_span_mappings.assert_called_once()
        mock_get_span_pairs.assert_called_once()

    def test_get_span_mappings(self):
        # TODO
        pass

    def test_get_spain_pairs(self):
        # TODO
        pass

    def test_relevance_score(self):
        # TODO
        pass

    @patch.object(PageEvaluation, "_get_span_pairs")
    @patch.object(PageEvaluation, "_get_span_mappings")
    @patch.object(PageEvaluation, "relevance_score", 0.5)
    def test_precision(self, mock_span_mapping, mock_span_pairs):
        mock_span_mapping.return_value = "", ""
        mock_span_pairs.return_value = ""
        # Edge case: No system spans
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=[])
        ## With reference spans
        page_ref_spans = NonCallableMock(page_id="id", spans=["s"])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.precision == 0
        ## No reference spans
        page_ref_spans = NonCallableMock(page_id="id", spans=[])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.precision == 1

        # With system spans
        sys_to_refs = [[], [0]]
        mock_span_mapping.return_value = "", sys_to_refs
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=["s"] * 2)
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.precision == 0.5 / 2
        sys_to_refs = [[], [1, 2], [3], [4, 5, 6]]
        mock_span_mapping.return_value = "", sys_to_refs
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=["s"] * 4)
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.precision == 0.5 / 7

    @patch.object(PageEvaluation, "_get_span_pairs")
    @patch.object(PageEvaluation, "_get_span_mappings")
    @patch.object(PageEvaluation, "relevance_score", 0.5)
    def test_recall(self, mock_span_mapping, mock_span_pairs):
        mock_span_mapping.return_value = "", ""
        mock_span_pairs.return_value = ""

        # Edge case: No reference spans
        page_ref_spans = NonCallableMock(page_id="id", spans=[])
        ## With system spans
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=["s"])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.recall == 0
        ## No system spans
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=[])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.recall == 1

        # With reference spans
        page_ref_spans = NonCallableMock(page_id="id", spans=["a"])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.recall == 0.5 / 1
        page_ref_spans = NonCallableMock(page_id="id", spans=["a"] * 3)
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.recall == 0.5 / 3
