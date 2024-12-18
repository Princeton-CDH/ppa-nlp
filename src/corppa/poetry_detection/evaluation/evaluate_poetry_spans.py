"""
Evaluate the poetry spans detected by a system against a provided reference set.
"""

from functools import cached_property


class Span:
    """
    Span object representing a Pythonic "closed open" interval.
    """

    def __init__(self, start: int, end: int, label: str):
        if end <= start:
            raise ValueError("Start index must be less than end index")
        self.start = start
        self.end = end
        self.label = label

    def __len__(self) -> int:
        return self.end - self.start

    def __eq__(self, other: "Span") -> bool:
        return (
            self.start == other.start
            and self.end == other.end
            and self.label == other.label
        )

    def __repr__(self) -> str:
        return f"Span({self.start}, {self.end}, {self.label})"

    def has_overlap(self, other: "Span", ignore_label: bool = False) -> bool:
        """
        Returns whether this span overlaps with the other span.
        Optionally, span labels can be ignored.
        """
        if ignore_label or self.label == other.label:
            return self.start < other.end and other.start < self.end
        return False

    def is_exact_match(self, other: "Span", ignore_label: bool = False) -> bool:
        """
        Checks if the other span is an exact match. Optionally, ignores
        span labels.
        """
        if self.start == other.start and self.end == other.end:
            return ignore_label or self.label == other.label
        else:
            return False

    def overlap_length(self, other: "Span", ignore_label: bool = False) -> int:
        """
        Returns the length of overlap between this span and the other span.
        Optionally, span labels can be ignored for this calculation.
        """
        if not self.has_overlap(other, ignore_label=ignore_label):
            return 0
        else:
            return min(self.end, other.end) - max(self.start, other.start)

    def overlap_factor(self, other: "Span", ignore_label: bool = False) -> float:
        """
        Returns the overlap factor with the other span. Optionally, span
        labels can be ignored for this calculation.

        The overlap factor is defined as follows:

            * If no overlap (overlap = 0), then overlap_factor = 0.

            * Otherwise, overlap_factor = overlap_length / longer_span_length

        So, the overlap factor has a range between 0 and 1 with higher values
        corresponding to a higher degree of overlap.
        """
        if not self.has_overlap(other, ignore_label=ignore_label):
            return 0
        else:
            overlap = self.overlap_length(other, ignore_label=ignore_label)
            return overlap / max(len(self), len(other))


class PageReferenceSpans:
    """
    Page-level reference spans object
    """

    def __init__(self, page_json):
        self.page_id = page_json["page_id"]
        self.spans = self._get_spans(page_json)

    def _get_spans(self, page_json):
        """
        Get span list from page-level json
        """
        spans = []
        if page_json["n_excerpts"] > 0:
            for excerpt in page_json["excerpts"]:
                spans.append(Span(excerpt["start"], excerpt["end"], excerpt["poem_id"]))

        # Sort spans by primarily by start index and secondarily by end index
        spans.sort(key=lambda x: (x.start, x.end))
        return spans


class PageSystemSpans:
    """
    Page-level system spans object.

    Note: Unlike a page's reference spans, the spans produced by a system might overlap.
    might overlap. While these overlapping spans are worth penalizing when span labels are
    taken in to account, this seems less true when labels are ignored.
    """

    def __init__(self, page_json):
        self.page_id = page_json["page_id"]
        self.labeled_spans = self._get_labeled_spans(page_json)
        self.unlabeled_spans = self._get_unlabeled_spans()

    def _get_labeled_spans(self, page_json):
        """
        Get (labeled) spans from page-level json
        """
        spans = []
        if page_json["n_spans"] > 0:
            for span in page_json["poem_spans"]:
                spans.append(Span(span["page_start"], span["page_end"], span["ref_id"]))
        # Sort spans by primarily by start index and secondarily by end index
        spans.sort(key=lambda x: (x.start, x.end))
        return spans

    def _get_unlabeled_spans(self):
        """
        Get "label-free" spans derived from object's labeled spans. All overlapping
        (but not adjacent) spans are merged into a single span.
        """
        spans = []
        for i, labeled_span in enumerate(self.labeled_spans):
            unlabeled_span = Span(labeled_span.start, labeled_span.end, "")
            if not i:
                # Edge case: Add the unlabeled version of the first span
                spans.append(unlabeled_span)
            else:
                # If there is overlap with the previous span, merge.
                # Note: This relies on the ordering of labeled_spans
                prev_span = spans[-1]
                if prev_span.has_overlap(unlabeled_span):
                    prev_span.end = max(prev_span.end, unlabeled_span.end)
                else:
                    spans.append(unlabeled_span)
        return spans


class PageEvaluation:
    """
    Page-level span evaluation.
    """

    def __init__(
        self,
        page_ref_spans: PageReferenceSpans,
        page_sys_spans: PageSystemSpans,
        ignore_label: bool = False,
    ):
        if page_ref_spans.page_id != page_sys_spans.page_id:
            raise ValueError(
                "Reference and system spans must correspond to the same page"
            )
        # Save working input
        self.ignore_label = ignore_label
        self.ref_spans = page_ref_spans.spans
        if self.ignore_label:
            self.sys_spans = page_sys_spans.unlabeled_spans
        else:
            self.sys_spans = page_sys_spans.labeled_spans
        # Determine mappings between reference and system spans
        self.ref_to_sys, self.sys_to_refs = self._get_span_mappings()
        # Determine the reference-system span pairs
        self.span_pairs = self._get_span_pairs()

    def _get_span_mappings(self):
        """
        Determines the mappings between the reference and system spans. Each
        reference span is mapped to at most one system span. Namely, the system
        span with the highest degree of overlap. Conversely, each system span
        is mapped to any number of overlapping (but disjoint) reference spans.
        """
        ref_to_sys = [None for _ in self.ref_spans]
        sys_to_refs = [[] for _ in self.sys_spans]

        # Assign each reference span to at most one system span
        for i, ref_span in enumerate(self.ref_spans):
            current_match_idx = None
            current_match_overlap = 0
            # Note: given these have an ordering this could be optimized
            for j, sys_span in enumerate(self.sys_spans):
                overlap = ref_span.overlap_factor(
                    sys_span, ignore_label=self.ignore_label
                )
                if overlap > current_match_overlap:
                    current_match_idx = j
                    current_match_overlap = overlap
            # Update mappings if a match is found
            if current_match_idx is not None:
                ref_to_sys[i] = current_match_idx
                sys_to_refs[j].append(i)
        return ref_to_sys, sys_to_refs

    def _get_span_pairs(self):
        """
        Determines the reference-system span pairs for evaluation using the
        existing mappings between reference and system pairs (`ref_to_sys`,
        `sys_to_refs`). A span pair corresponds is a tuple of
            1. A reference span r mapped to system span s
            2. The system span s if s solely maps to r. Otherwise, a subspan of s.

        When a system span s maps to k > 1 reference spans r_i. It's split into
        subspans with the following ranges:
            (s start, r_1 start), (r_1 start, r_2 start), ..., (r_k start, s end)
        """
        span_pairs = []
        for sys_id, ref_ids in self.sys_to_refs.items():
            sys_span = self.sys_spans[sys_id]
            if len(ref_ids) == 1:
                ref_span = self.ref_spans[ref_ids[0]]
                span_pairs.append(ref_span, sys_span)
            else:
                # Effectively, split system span into k pieces (one for each reference span)
                for i, ref_id in enumerate(ref_ids):
                    ref_span = self.ref_spans[ref_id]
                    start = ref_span.start if i else sys_span.start
                    end = ref_span.end if i + 1 < len(ref_ids) else sys_span.end
                    sub_sys_span = Span(start, end, sys_span.label)
                    span_pairs.append(ref_span, sub_sys_span)
        return span_pairs

    @cached_property
    def relevance_score(self, partial_match_weight=1) -> float:
        """
        Computes the relevance score used for calculating precision and recall.
        Optionally can specify a downweight for partial matches.
        """
        score = 0
        for ref_span, sys_span in self.span_pairs:
            if ref_span.is_exact_match(sys_span, ignore_label=self.ignore_label):
                score += 1
            else:
                score += partial_match_weight * ref_span.overlap_factor(sys_span)
        return score

    @cached_property
    def precision(self) -> float:
        """
        Calculate page-level precision. Edge case: If there are no system spans,
        return 1 if there are also no reference spans and 0 otherwise.
        """
        if not self.sys_spans:
            # Edge case to avoid divide by zero error
            return 1 if not self.ref_spans else 0

        # Determine number of retrieved (possibly split) system spans.
        n_retrieved = 0
        for ref_ids in self.sys_to_refs:
            if ref_ids:
                n_retrieved += len(ref_ids)
            else:
                # Include spurious/incorrect spans
                n_retrieved += 1
        return self.relevance_score / n_retrieved

    @cached_property
    def recall(self) -> float:
        """
        Calculate page-level recall. Edge case: If there are no reference spans,
        return 1 if there are also no system spans and 0 otherwise.
        """
        if not self.ref_spans:
            # Edge case to avoid divide by zero error
            return 1 if not self.sys_spans else 0

        n_relevant = len(self.ref_spans)
        return self.relevance_score / n_relevant


def get_page_span_evaluation():
    raise NotImplementedError


def get_page_evals(ref_file, sys_file):
    raise NotImplementedError


def write_page_evals(ref_file, sys_file, out_file):
    raise NotImplementedError
