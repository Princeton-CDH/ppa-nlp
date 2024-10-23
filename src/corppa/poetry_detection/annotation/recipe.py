"""
This module provides custom recipes for Prodigy annotation. These were
created with page-level annotation in mind, and assume a page is associated
with both an image and text.

Recipes:
    * `annotate_page_text`: Annotate a page's text with the page's image
      displayed side-by-side for reference.
    * `annotate_text_and_image`: Annotate both a page's text and image side-by-side.

Referenced images must be served out independently for display; the image url
prefix for images should be specified when initializing the recipe.

Example use:
```
prodigy annotate_page_text poetry_spans poetry_pages.jsonl --label POETRY,PROSODY -F ../corppa/poetry_detection/annotation/recipe.py --image-prefix http://localhost:8000/
prodigy annotate_text_and_image poetry_text_image poetry_pages.jsonl -l POETRY -F ../corppa/poetry_detection/annotation/recipe.py --image-prefix http://localhost:8000/
"""

from pathlib import Path

import spacy
from prodigy import log
from prodigy.components.preprocess import add_tokens
from prodigy.components.preprocess import fetch_media as fetch_media_preprocessor
from prodigy.components.stream import get_stream
from prodigy.core import Arg, recipe
from prodigy.util import get_labels

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


def add_image(task, image_prefix=None):
    if image_prefix is None:
        task["image"] = task["image_path"]
    else:
        path_pfx = image_prefix.rstrip("/")
        task["image"] = f"{path_pfx}/{task['image_path']}"
    return task


def add_images(examples, image_prefix=None):
    for task in examples:
        yield add_image(task, image_prefix=image_prefix)


def remove_images(examples, image_prefix=None):
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
    labels: str,
    image_prefix: str = None,
    fetch_media: bool = False,
):
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
        stream.apply(fetch_media_preprocessor, ["image"], skip=True)

    # split labels by commas and strip any whitespace
    label_list = get_labels(labels)

    blocks = [
        {
            "view_id": "image_manual",
            "labels": label_list,
        },
        {"view_id": "spans_manual", "labels": label_list},
    ]

    # copy the common config options and add blocks and labels
    config = PRODIGY_COMMON_CONFIG.copy()
    config.update(
        {
            "blocks": blocks,
            "labels": label_list,
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
    labels: str,
    image_prefix: str = None,
    fetch_media: bool = False,
):
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
        stream.apply(fetch_media_preprocessor, ["image"], skip=True)

    # split labels by commas and strip any whitespace
    label_list = get_labels(labels)

    blocks = [
        {
            "view_id": "html",
            "html_template": "<img src='{{ image }}' width='500'>",
        },
        {"view_id": "spans_manual", "labels": label_list},
    ]
    # copy the common config options and add blocks and labels
    config = PRODIGY_COMMON_CONFIG.copy()
    config.update(
        {
            "blocks": blocks,
            "labels": label_list,
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
