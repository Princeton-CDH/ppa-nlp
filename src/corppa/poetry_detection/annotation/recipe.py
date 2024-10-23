"""
This module provides custom recipes for Prodigy annotation. These were
created with page-level annotation in mind and assume a page is associated
with both text and an image. Each recipe displays a page's image and text
side-by-side.

Recipes:
    * `annotate_page_text`: Annotate a page's text.
    * `annotate_text_and_image`: Annotate both a page's text and image side-by-side.
    * `review_page_spans`: Review existing page-level text annotations to produce
      a final, adjudicated set of annotations.

Referenced images must be served out independently for display; the image url
prefix for images should be specified when initializing the recipe.

Example use:
```
prodigy annotate_page_text poetry_spans poetry_pages.jsonl --label POETRY,PROSODY -F ../corppa/poetry_detection/annotation/recipe.py --image-prefix http://localhost:8000/
prodigy annotate_text_and_image poetry_text_image poetry_pages.jsonl -l POETRY -F ../corppa/poetry_detection/annotation/recipe.py --image-prefix ../ppa-web-images -FM
prodigy review_page_spans adjudicate poetry_spans -l POETRY -F ../corppa/poetry_detection/annotation/recipe.py --image-prefix ../ppa-web-images -FM --sessions alice,bob
"""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import spacy
from prodigy import log, set_hashes
from prodigy.components.db import connect
from prodigy.components.preprocess import add_tokens
from prodigy.components.preprocess import fetch_media as fetch_media_preprocessor
from prodigy.components.stream import get_stream
from prodigy.core import Arg, recipe
from prodigy.types import LabelsType, RecipeSettingsType, StreamType, TaskType
from prodigy.util import INPUT_HASH_ATTR, SESSION_ID_ATTR, get_labels

#: reference to current directory, for use as Prodigy CSS directory
CURRENT_DIR = Path(__file__).parent.absolute()

#: common prodigy configurations for both recipes; copy and add blocks and labels
PRODIGY_COMMON_CONFIG = {
    "buttons": ["accept", "reject", "undo"],  # remove ignore button
    "show_flag": True,  # show flag button to mark weird/difficult examples
    "hide_newlines": False,  # ensure newlines are shown \n
    "allow_newline_highlight": True,  # allow highlighting of newlines \n
    "honor_token_whitespace": True,  # reflect whitespace accurately (e.g. in case of leading/trailing spaces)
    "custom_theme": {
        "labels": {
            # trying to use options from PPA webapp color scheme,
            # but may not be so great in Prodigy UI.
            # azure #0788fc seafoam blue #57c4c4 wisteria #9c93c0 pig pink #ed949c
            "POETRY": "#57c4c4",  # label color for POETRY
        },
        "hide_true_newline_tokens": False,
    },
    "global_css_dir": CURRENT_DIR,
}

#: color palette for predefined session names
PALETTE = [
    "#c5bdf4",
    "#ffd882",
    "#d9fbad",
    "#c2f2f6",
    "#ffdaf9",
    "#b5c6c9",
    "#96e8ce",
    "#ffd1b2",
]


def add_image(example: TaskType, image_prefix: Optional[str] = None):
    """
    Set an example's image field to its existing image_path with an optional prefix

    Note: Assumes filepaths use forward slash
    """
    if image_prefix is None:
        example["image"] = example["image_path"]
    else:
        path_pfx = image_prefix.rstrip("/")
        example["image"] = f"{path_pfx}/{example['image_path']}"
    return example


def add_images(examples: StreamType, image_prefix: Optional[str] = None) -> StreamType:
    """
    Set the image field for each example in the stream

    Calls: `add_image`
    """
    for example in examples:
        yield add_image(example, image_prefix=image_prefix)


def remove_images(
    examples: Iterable[TaskType], image_prefix: Optional[str] = None
) -> List[TaskType]:
    """
    For each example, replace base64 data URIs with image filepath or URL

    Calls: `add_image`
    """
    for task in examples:
        # If "image" is a base64 string and "image_path" is present in the task,
        # remove the image data
        if task["image"].startswith("data:") and "image_path" in task:
            # Replace image with full image path
            add_image(task, image_prefix=image_prefix)
    return examples


@recipe(
    "annotate_text_and_image",
    dataset=Arg(help="path to input dataset"),
    labels=Arg(
        "--label",
        "-l",
        help="Comma-separated label(s) to annotate or text file with one label per line",
    ),
    image_prefix=Arg("--image-prefix", "-i", help="Base URL for images"),
    fetch_media=Arg(
        "--fetch-media", "-FM", help="Load images from local paths or URLs"
    ),
)
def annotate_text_and_image(
    dataset: str,
    source: str,
    labels: LabelsType = [],
    image_prefix: str = None,
    fetch_media: bool = False,
) -> RecipeSettingsType:
    """Annotate text and image side by side: allows adding manual spans
    to both image and text. Intended for page-level annotation.
    """
    log("RECIPE: Starting recipe annotate_text_and_image", locals())
    stream = get_stream(source)
    # add tokens tokenize
    stream.apply(add_tokens, nlp=spacy.blank("en"), stream=stream)
    # add image prefix
    stream.apply(add_images, image_prefix=image_prefix)
    # optionally fetch media
    if fetch_media:
        stream.apply(fetch_media_preprocessor, ["image"])

    blocks = [
        {
            "view_id": "image_manual",
            "labels": labels,
        },
        {"view_id": "spans_manual", "labels": labels},
    ]

    # copy the common config options and add blocks and labels
    config = deepcopy(PRODIGY_COMMON_CONFIG)
    config.update(
        {
            "blocks": blocks,
            "ner_manual_highlight_chars": True,
            "image_manual_spans_key": "image_spans",
            # limit image selection to rectangle only, no polygon or freehand
            "image_manual_modes": ["rect"],
        }
    )

    components = {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": config,
    }

    if fetch_media:
        components["before_db"] = lambda x: remove_images(x, image_prefix=image_prefix)

    return components


@recipe(
    "annotate_page_text",
    dataset=Arg(help="path to input dataset"),
    labels=Arg(
        "--label",
        "-l",
        help="Comma-separated label(s) to annotate or text file with one label per line",
    ),
    image_prefix=Arg("--image-prefix", "-i", help="Base URL for images"),
    fetch_media=Arg(
        "--fetch-media", "-FM", help="Load images from local paths or URLs"
    ),
)
def annotate_page_text(
    dataset: str,
    source: str,
    labels: LabelsType = [],
    image_prefix: str = None,
    fetch_media: bool = False,
) -> RecipeSettingsType:
    """Annotate text with manual spans; displays an image side by side
    with text for reference only (image cannot be annotated).
    Intended for page-level annotation.
    """
    log("RECIPE: Starting recipe annotate_page_text", locals())
    stream = get_stream(source)
    # add tokens tokenize
    stream.apply(add_tokens, nlp=spacy.blank("en"), stream=stream)
    # add image prefix
    stream.apply(add_images, stream, image_prefix=image_prefix)
    # optionally fetch media
    if fetch_media:
        stream.apply(fetch_media_preprocessor, ["image"])

    blocks = [
        {
            "view_id": "html",
            "html_template": "<img src='{{ image }}' width='500'>",
        },
        {"view_id": "spans_manual", "labels": labels},
    ]
    # copy the common config options and add blocks and labels
    config = deepcopy(PRODIGY_COMMON_CONFIG)
    config.update(
        {
            "blocks": blocks,
            "ner_manual_highlight_chars": True,
        }
    )

    components = {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": config,
    }

    if fetch_media:
        components["before_db"] = lambda x: remove_images(x, image_prefix=image_prefix)

    return components


class ReviewStream:
    """
    Stream of review examples. This mostly exists to expose a __len__ to show
    total progress in the web interface.
    """

    def __init__(
        self,
        data: Dict[int, List[TaskType]],
        image_prefix: Optional[str] = None,
        fetch_media: bool = False,
    ) -> None:
        """
        Initialize a review stream.

        data: Merged data, with examples grouped by input hash.
        image_prefix: Image prefix for creating image (full) paths
        fetch_media: Whether to fetch task images.
        """
        self.n_examples = len(data)
        self.data = self.get_data(data, image_prefix, fetch_media)

    def __len__(self) -> int:
        return self.n_examples

    def __iter__(self) -> StreamType:
        for example in self.data:
            yield example

    def create_review_example(self, versions: List[TaskType]) -> TaskType:
        """
        Create review example from several annotated versions.
        """
        # TODO: Make sure that no unmerged version content is preserved.
        review_example = deepcopy(versions[-1])
        # Merge spans
        merged_spans = []
        session_counts = {}
        sessions = []

        for version in versions:
            session_id = version[SESSION_ID_ATTR]
            # Assume: session name does not contain -
            # full session name includes the dataset id; split to get the session name without dataset id
            session_name = session_id.rsplit("-", maxsplit=1)[1]
            if session_id not in session_counts:
                session_counts[session_id] = 1
            else:
                session_name += f"-{session_counts[session_id]}"
                session_counts[session_id] += 1
            sessions.append(session_name)
            if "spans" not in version:
                # Not sure when an annotated example would be missing a spans field
                continue
            for span in version["spans"]:
                new_span = span.copy()
                span_label = span["label"]
                new_span["label"] = f"{session_name}: {span_label}"
                merged_spans.append(new_span)
        review_example["spans"] = merged_spans
        review_example["sessions"] = sessions
        return review_example

    def get_data(
        self,
        data: Dict[int, List[TaskType]],
        image_prefix: Optional[str],
        fetch_media: bool,
    ) -> List[TaskType]:
        """
        Build review examples from data. Add images to each example.
        """
        examples = []
        for _, versions in data.items():
            review_example = self.create_review_example(versions)
            review_example = add_image(review_example, image_prefix)
            examples.append(review_example)
        if fetch_media:
            return fetch_media_preprocessor(examples, ["image"])
        return examples


def get_review_stream(
    examples: Iterable[TaskType],
    image_prefix: Optional[str] = None,
    fetch_media: bool = False,
) -> StreamType:
    # Group examples by input (page_id, text)
    grouped_examples = defaultdict(list)
    for example in examples:
        # Reset hashes
        example = set_hashes(
            example, overwrite=True, input_keys=["id", "text"], task_keys=["spans"]
        )
        input_hash = example[INPUT_HASH_ATTR]
        grouped_examples[input_hash].append(example)
    return ReviewStream(
        grouped_examples, image_prefix=image_prefix, fetch_media=fetch_media
    )


@recipe(
    "review_page_spans",
    dataset=Arg(help="Dataset to save annotations to"),
    input_dataset=Arg(help="Name of dataset to review"),
    labels=Arg(
        "--label",
        "-l",
        help="Comma-separated label(s) to annotate or text file with one label per line",
    ),
    image_prefix=Arg("--image-prefix", "-i", help="Base URL for images"),
    fetch_media=Arg(
        "--fetch-media", "-FM", help="Load images from local paths or URLs"
    ),
    sessions=Arg("--sessions", help="Comma-separated session names for coloring"),
)
def review_page_spans(
    dataset: str,
    input_dataset: str,
    labels: LabelsType = [],
    image_prefix: str = None,
    fetch_media: bool = False,
    sessions: List[str] = [],
) -> RecipeSettingsType:
    """
    Review input text span annotations and annotate with manual spans to create
    final, adjudicated annotations. Loads and displays input text span
    annotations.
    """
    # Load annotations
    DB = connect()
    if input_dataset not in DB:
        raise RecipeError(f"Can't find input dataset '{input_dataset}' in database")
    annotations = DB.get_dataset_examples(input_dataset)

    blocks = [
        {
            "view_id": "html",
            "html_template": "<img src='{{ image }}' width='500'>",
        },
        {"view_id": "spans_manual", "labels": labels},
    ]

    def before_db(examples):
        """
        Modifies annotated examples before saving to the database:
            * Remove image spans & tokens (unneeded fields)
            * Reset image to (full) image path if image fetched
        """
        for example in examples:
            # remove image spans
            del example["image_spans"]
            # remove tokens
            del example["tokens"]
            if fetch_media:
                # reset image to path
                example = add_image(example, image_prefix=image_prefix)
        return examples

    # Set label colors
    label_colors = PRODIGY_COMMON_CONFIG["custom_theme"]["labels"].copy()
    if sessions:
        # Add session-label colors
        for i, session in enumerate(sessions):
            session_color = PALETTE[i % len(PALETTE)]
            for label in labels:
                label_colors[f"{session}: {label}"] = session_color

    # copy the common config options and add blocks and labels
    config = deepcopy(PRODIGY_COMMON_CONFIG)
    config.update(
        {
            "blocks": blocks,
            "ner_manual_highlight_chars": True,
            "global_css_dir": CURRENT_DIR,
            "custom_theme": {"labels": label_colors},
        }
    )

    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": get_review_stream(
            annotations, image_prefix=image_prefix, fetch_media=fetch_media
        ),
        "before_db": before_db,
        "config": config,
    }
