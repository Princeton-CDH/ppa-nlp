"""
This module provides custom recipes for Prodigy annotation. They were
created with page-level text annotation in mind, and support annotating
text with a reference image displayed beside the text (`annotate_page_text`),
or annotating both text and image side by side (`annotate_text_and_image`).

Referenced images must be served out independently for display; the image url
prefix for images should be specified when initializing the recipe.

Example use:
```
prodigy annotate_page_text poetry_spans poetry_pages.jsonl --label POETRY,PROSODY -F ../corppa/poetry_detection/annotation/recipe.py --image-prefix http://localhost:8000/
prodigy annotate_text_and_image poetry_text_image poetry_pages.jsonl --label POETRY -F ../corppa/poetry_detection/annotation/recipe.py --image-prefix http://localhost:8000/
"""

from pathlib import Path

import spacy
from prodigy.components.loaders import JSONL
from prodigy.core import Arg, recipe

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
    # Task annotation flags, these should be move elsewhere
    "feed_overlap": False,
    "annotations_per_task": 2,
}


def tokenize_stream(stream, image_prefix=None):
    """Takes a stream of Prodigy tasks and tokenizes text for span annotation,
    and optionally adds an image prefix URL to any image paths present.
    Stream is expected to contain `text` and may contain image_path` and a `meta`
    dictionary. Returns a generator of the stream.
    """

    nlp = spacy.blank("en")  # use blank spaCy model for tokenization

    # ensure image prefix URL does not have a trailing slash
    if image_prefix is None:
        image_prefix = ""
    image_prefix = image_prefix.rstrip("/")

    for task in stream:
        if task.get("text"):
            doc = nlp(task["text"])
            task["tokens"] = [
                {
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "id": i,
                }
                for i, token in enumerate(doc)
            ]
        # add image prefix URL for serving out images
        if "image_path" in task:
            task["image"] = f"{image_prefix}/{task['image_path']}"
        yield task


@recipe(
    "annotate_text_and_image",
    dataset=Arg(help="path to input dataset"),
    labels=Arg("--label", "-l", help="Comma-separated label(s)"),
    image_prefix=Arg("--image-prefix", "-i", help="Base URL for images"),
)
def annotate_text_and_image(
    dataset: str, source: str, labels: str, image_prefix: str = None
):
    """Annotate text and image side by side: allows adding manual spans
    to both image and text. Intended for page-level annotation.
    """

    stream = JSONL(source)  # load jsonlines into stream
    # tokenize for span annotation and add image prefix
    tokenized_stream = tokenize_stream(stream, image_prefix)

    # split labels by commas and strip any whitespace
    label_list = [label.strip() for label in labels.split(",")]

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

    return {
        "dataset": dataset,
        "stream": tokenized_stream,
        "view_id": "blocks",
        "config": config,
    }


@recipe(
    "annotate_page_text",
    dataset=Arg(help="path to input dataset"),
    labels=Arg("--label", "-l", help="Comma-separated label(s)"),
    image_prefix=Arg("--image-prefix", "-i", help="Base URL for images"),
)
def annotate_page_text(
    dataset: str, source: str, labels: str, image_prefix: str = None
):
    """Annotate text with manual spans; displays an image side by side
    with text for reference only (image cannot be annotated).
    Intended for page-level annotation.
    """

    stream = JSONL(source)  # load jsonlines into stream
    # tokenize for span annotation and add image prefix
    tokenized_stream = tokenize_stream(stream, image_prefix)

    # split labels by commas and strip any whitespace
    label_list = [label.strip() for label in labels.split(",")]

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

    return {
        "dataset": dataset,
        "stream": tokenized_stream,
        "view_id": "blocks",
        "config": config,
    }
