import sys
from collections import defaultdict
from unittest.mock import MagicMock, call, patch

import pytest

# Skip spacy & prodigy dependencies not needed for testing
skip_deps = [
    "spacy",
    "prodigy.components.db",
    "prodigy.components.stream",
    "prodigy.core",
    "prodigy.errors",
    "prodigy.types",
    "prodigy.util",
]
for dep in skip_deps:
    sys.modules[dep] = MagicMock()
# Mock prodigy dependencies used in testing
## Mock for set_hashes
mock_prodigy = MagicMock()
sys.modules["prodigy"] = mock_prodigy
## Mock for fetch_media
mock_prodigy_preprocess = MagicMock()
sys.modules["prodigy.components.preprocess"] = mock_prodigy_preprocess

from corppa.poetry_detection.annotation.annotation_recipes import (
    ReviewStream,
    add_image,
    add_images,
    add_label_prefix,
    add_session_prefix,
    get_review_stream,
    get_session_name,
    has_span_overlap,
    remove_image_data,
    remove_label_prefix,
    remove_session_prefix,
    validate_review_answer,
)


def test_add_image():
    example = {}

    with pytest.raises(ValueError, match="Cannot add image, image_path missing"):
        add_image(example)

    example["image_path"] = "some/image/path"

    # Call without prefix
    assert add_image(example) is example
    assert example["image"] == example["image_path"]

    # Call with prefix
    assert add_image(example, image_prefix="prefix") is example
    assert example["image"] == f"prefix/{example['image_path']}"


@patch("corppa.poetry_detection.annotation.annotation_recipes.add_image")
def test_add_images(mock_add_image):
    examples = [{"image_path": "a"}, {"image_path": "b"}]

    results = list(add_images(examples))
    assert mock_add_image.call_count == 2
    mock_add_image.assert_has_calls([call(ex, image_prefix=None) for ex in examples])

    examples.append({"image_path": "c"})
    mock_add_image.reset_mock()
    results = list(add_images(examples, image_prefix="prefix"))
    assert mock_add_image.call_count == 3
    mock_add_image.assert_has_calls(
        [call(ex, image_prefix="prefix") for ex in examples]
    )


@patch("corppa.poetry_detection.annotation.annotation_recipes.add_image")
def test_remove_image_data(mock_add_image):
    # Empty list (i.e. no examples)
    assert remove_image_data([]) == []
    mock_add_image.assert_not_called()
    assert remove_image_data([], image_prefix="pfx") == []
    mock_add_image.assert_not_called()

    # Examples where image data should not be removed
    examples_without_data = [
        {},  # empty example
        {"image": "data:uri"},  # no image_path
        {"image": "some/path", "image_path": "some/path"},  # image w/o "base:" prefix
    ]
    assert remove_image_data(examples_without_data) is examples_without_data
    mock_add_image.assert_not_called()

    # Examples with image data to be removed
    examples_with_data = [
        {"image": "data:string+a", "image_path": "path/a"},
        {"image": "data:string+b", "image_path": "path/b"},
    ]
    assert remove_image_data(examples_with_data) is examples_with_data
    assert mock_add_image.call_count == 2
    mock_add_image.assert_has_calls(
        [call(ex, image_prefix=None) for ex in examples_with_data]
    )

    # Some examples should be removed
    mock_add_image.reset_mock()
    mixed_examples = [
        {"image": "some/path", "image_path": "some/path"},  # image w/o "base:" prefix
        {"image": "data:string+c", "image_path": "path/c"},
    ]
    assert remove_image_data(mixed_examples, image_prefix="prefix") is mixed_examples
    mock_add_image.assert_called_once()
    assert mock_add_image.call_args == call(mixed_examples[-1], image_prefix="prefix")


@patch(
    "corppa.poetry_detection.annotation.annotation_recipes.SESSION_ID_ATTR",
    "session_id",
)
def test_get_session_name():
    # Typical case: drop db prefix
    example = {"session_id": "db-id-alice"}
    assert get_session_name(example) == "alice"
    assert get_session_name(example, 1) == "alice-1"

    # Atypical case: session id has no prefix
    example = {"session_id": "test"}
    assert get_session_name(example) == "test"
    assert get_session_name(example, suffix="sfx") == "test-sfx"


def test_add_label_prefix():
    assert add_label_prefix("label", "prefix") == "prefix: label"
    assert add_label_prefix("label", "") == ": label"


def test_remove_label_prefix():
    assert remove_label_prefix("pfx: label") == "label"
    assert remove_label_prefix("no_prefix") == "no_prefix"


@patch("corppa.poetry_detection.annotation.annotation_recipes.get_session_name")
@patch("corppa.poetry_detection.annotation.annotation_recipes.add_label_prefix")
def test_add_session_prefix(mock_add_label_prefix, mock_get_session_name):
    mock_get_session_name.return_value = "session"

    # example without spans
    example = {"text": "some text..."}
    assert add_session_prefix(example) is example
    assert example == {"text": "some text..."}  # example is unchanged
    mock_get_session_name.assert_called_once()
    mock_add_label_prefix.assert_not_called()

    # example with spans
    mock_get_session_name.reset_mock()
    example = {"spans": [{"label": "a"}, {"label": "b"}]}
    assert add_session_prefix(example) is example
    mock_get_session_name.assert_called_once()
    assert mock_add_label_prefix.call_count == 2
    mock_add_label_prefix.assert_has_calls([call("a", "session"), call("b", "session")])


@patch("corppa.poetry_detection.annotation.annotation_recipes.remove_label_prefix")
def test_remove_session_prefix(mock_remove_label_prefix):
    # example without spans
    example = {"text": "some text..."}
    assert remove_session_prefix(example) is example
    assert example == {"text": "some text..."}  # example is unchanged
    mock_remove_label_prefix.assert_not_called()

    # example with spans
    example = {"spans": [{"label": "a"}, {"label": "b"}]}
    assert remove_session_prefix(example) is example
    assert mock_remove_label_prefix.call_count == 2
    mock_remove_label_prefix.assert_has_calls([call("a"), call("b")])


@patch("corppa.poetry_detection.annotation.annotation_recipes.remove_label_prefix")
def test_has_span_overlap(mock_remove_label_prefix):
    # example without spans
    ex_no_spans = {}
    assert has_span_overlap(ex_no_spans) is False
    mock_remove_label_prefix.assert_not_called()

    # examples without overlapping spans
    mock_remove_label_prefix.side_effect = lambda x: x
    ex_consecutive = {
        "spans": [
            {"start": 0, "end": 4, "label": "a"},
            {"start": 4, "end": 8, "label": "a"},
            {"start": 8, "end": 9, "label": "a"},
        ]
    }
    ex_distinct_labels = {
        "spans": [
            {"start": 5, "end": 7, "label": "b"},
            {"start": 5, "end": 7, "label": "c"},
        ]
    }
    assert has_span_overlap(ex_consecutive) is False
    assert mock_remove_label_prefix.call_count == 3
    mock_remove_label_prefix.assert_has_calls([call("a"), call("a"), call("a")])
    assert has_span_overlap(ex_distinct_labels) is False
    assert mock_remove_label_prefix.call_count == 5
    mock_remove_label_prefix.assert_has_calls([call("b"), call("c")])

    # example with overlap
    mock_remove_label_prefix.side_effect = lambda x: x[0]
    ex_with_overlap = {
        "spans": [
            {"start": 3, "end": 8, "label": "a1"},
            {"start": 7, "end": 9, "label": "a2"},
            {"start": 1, "end": 2, "label": "a3"},
        ]
    }
    ex_with_nested = {
        "spans": [
            {"start": 1, "end": 5, "label": "b1"},
            {"start": 2, "end": 3, "label": "b2"},
        ]
    }
    assert has_span_overlap(ex_with_overlap) is True
    assert has_span_overlap(ex_with_nested) is True

    # Test disabling label prefix stripping
    mock_remove_label_prefix.reset_mock()
    assert has_span_overlap(ex_consecutive, strip_label_pfx=False) is False
    assert has_span_overlap(ex_with_overlap, strip_label_pfx=False) is False
    mock_remove_label_prefix.assert_not_called()


@patch("corppa.poetry_detection.annotation.annotation_recipes.has_span_overlap")
def test_validate_review_stream(mock_has_span_overlap):
    mock_has_span_overlap.side_effect = [False, True, False]

    # Success
    validate_review_answer({})
    mock_has_span_overlap.assert_called_once()

    # Failure: flagged example
    mock_has_span_overlap.reset_mock()
    with pytest.raises(ValueError, match="Currently flagged, unflag to submit."):
        validate_review_answer({"flagged": True})
    mock_has_span_overlap.assert_not_called()

    # Failure: overlapping spans
    with pytest.raises(
        ValueError, match="Overlapping spans with the same label detected!"
    ):
        validate_review_answer({})
    mock_has_span_overlap.assert_called_once()


class TestReviewStream:
    @patch.object(ReviewStream, "get_data")
    def test_init(self, mock_get_data):
        mock_get_data.return_value = "data"
        stream = ReviewStream({})
        assert stream.data == "data"
        mock_get_data.assert_called_once()
        assert mock_get_data.call_args == call({}, None, False)

        mock_get_data.reset_mock()
        data = {1: ["v1", "v2"]}
        stream = ReviewStream(data, image_prefix="pfx", fetch_media="fetch_media")
        mock_get_data.assert_called_once()
        assert mock_get_data.call_args == call(data, "pfx", "fetch_media")

    @patch("corppa.poetry_detection.annotation.annotation_recipes.add_session_prefix")
    @patch("corppa.poetry_detection.annotation.annotation_recipes.get_session_name")
    def test_create_review_example(
        self, mock_get_session_name, mock_add_session_prefix
    ):
        # Empty list
        with pytest.raises(
            ValueError,
            match="Cannot create review example without one or more annotated versions",
        ):
            ReviewStream.create_review_example([])
            mock_get_session_name.assert_not_called()
            mock_add_session_prefix.assert_not_called()

        # Single, minimal example
        mock_get_session_name.return_value = "session_name"
        versions = [{}]
        assert ReviewStream.create_review_example([{}]) == {
            "sessions": ["session_name"],
            "spans": [],
        }
        assert versions == [{}]  # unchanged (no spans to alter)
        mock_get_session_name.assert_called_once()
        mock_add_session_prefix.assert_called_once()

        # Multiple examples, distinct sessions
        mock_add_session_prefix.reset_mock()
        mock_get_session_name.reset_mock()
        mock_get_session_name.side_effect = ["a", "b", "c"]
        task_a = {"spans": [{"start": 0, "end": 4, "label": "1"}]}
        task_b = {"spans": [{"start": 0, "end": 3, "label": "2"}]}
        versions = [task_a, task_b, {}]
        result = ReviewStream.create_review_example(versions)
        assert result == {
            "spans": [task_a["spans"][0], task_b["spans"][0]],
            "sessions": ["a", "b", "c"],
        }
        assert mock_get_session_name.call_count == 3
        assert mock_add_session_prefix.call_count == 3
        mock_add_session_prefix.assert_has_calls([call(v) for v in versions])

        # Multiple examples, duplicate sessions
        mock_add_session_prefix.reset_mock()
        mock_get_session_name.reset_mock(side_effect=True)
        mock_get_session_name.return_value = "a"
        result = ReviewStream.create_review_example(versions)
        assert result == {
            "spans": [task_a["spans"][0], task_b["spans"][0]],
            "sessions": ["a"],
        }
        assert mock_get_session_name.call_count == 3
        assert mock_add_session_prefix.call_count == 3
        mock_add_session_prefix.assert_has_calls(
            [call(task_a), call(task_b, session_sfx=1), call({}, session_sfx=2)]
        )

        # Test flag behavior (set if any flagged example)
        task_a["flagged"] = True
        result = ReviewStream.create_review_example(versions)
        assert result == {
            "spans": [task_a["spans"][0], task_b["spans"][0]],
            "sessions": ["a"],
            "flagged": True,
        }

    @patch.object(ReviewStream, "create_review_example")
    @patch("corppa.poetry_detection.annotation.annotation_recipes.add_image")
    def test_get_data(self, mock_add_image, mock_create_review_example):
        mock_prodigy_preprocess.fetch_media.reset_mock(
            return_value=True, side_effect=True
        )
        mock_prodigy_preprocess.fetch_media.return_value = "examples_with_media"
        data = {1: ["a"], 2: ["b", "c"], 3: ["c", "b", "c"]}

        # Note: initialization calls get_data
        mock_create_review_example.side_effect = lambda x: x[0]
        mock_add_image.return_value = "with_image"
        stream = ReviewStream(data, image_prefix="pfx", fetch_media=False)
        assert len(stream) == 3
        assert stream.data == ["with_image"] * 3
        assert mock_create_review_example.call_count == 3
        mock_create_review_example.assert_has_calls(
            [call(val) for val in data.values()]
        )
        assert mock_add_image.call_count == 3
        mock_add_image.assert_has_calls(
            [call("a", "pfx"), call("b", "pfx"), call("c", "pfx")]
        )
        mock_prodigy_preprocess.fetch_media.assert_not_called()

        # Fetch media set
        mock_create_review_example.reset_mock(side_effect=True)
        mock_add_image.reset_mock()
        mock_create_review_example.return_value = "review"
        stream = ReviewStream(data, fetch_media=True)
        mock_prodigy_preprocess.fetch_media.assert_called_once()
        assert stream.data == "examples_with_media"
        mock_prodigy_preprocess.fetch_media.call_args == call(
            ["with_image"] * 3, ["image"]
        )
        assert mock_create_review_example.call_count == 3
        mock_create_review_example.assert_has_calls(
            [call(val) for val in data.values()]
        )
        assert mock_add_image.call_count == 3
        mock_add_image.assert_has_calls([call("review", None) for _ in range(3)])


@patch(
    "corppa.poetry_detection.annotation.annotation_recipes.INPUT_HASH_ATTR",
    "input_hash",
)
@patch("corppa.poetry_detection.annotation.annotation_recipes.ReviewStream")
def test_get_review_stream(mock_stream):
    mock_prodigy.set_hashes.reset_mock(return_value=True, side_effect=True)
    mock_prodigy.set_hashes.side_effect = lambda x, overwrite, input_keys, task_keys: x

    task_a = {"id": "a", "input_hash": 1}
    task_b = {"id": "b", "input_hash": 2}
    task_c = {"id": "c", "input_hash": 1}
    examples = [task_a, task_b, task_c]

    get_review_stream(examples)
    mock_stream.assert_called_once()
    assert mock_stream.call_args == call(
        defaultdict(list, {1: [task_a, task_c], 2: [task_b]}),
        image_prefix=None,
        fetch_media=False,
    )
    assert mock_prodigy.set_hashes.call_count == 3
    mock_prodigy.set_hashes.assert_has_calls(
        [
            call(ex, overwrite=True, input_keys=["id", "text"], task_keys=["spans"])
            for ex in examples
        ]
    )

    mock_stream.reset_mock()
    mock_prodigy.set_hashes.reset_mock()
    get_review_stream(examples, "prefix", "fetch_media")
    assert mock_stream.call_args == call(
        defaultdict(list, {1: [task_a, task_c], 2: [task_b]}),
        image_prefix="prefix",
        fetch_media="fetch_media",
    )
    assert mock_prodigy.set_hashes.call_count == 3
    mock_prodigy.set_hashes.assert_has_calls(
        [
            call(ex, overwrite=True, input_keys=["id", "text"], task_keys=["spans"])
            for ex in examples
        ]
    )
