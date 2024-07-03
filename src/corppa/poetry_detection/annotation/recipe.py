from prodigy.core import Arg, recipe
from prodigy.components.loaders import JSONL
import spacy

from pathlib import Path

CURRENT_DIR = Path(__file__).parent.absolute()


@recipe(
    "annotate_poetry_with_image",
    dataset=Arg(help="path to input dataset"),
    labels=Arg("--label", "-l", help="Comma-separated label(s)"),
    image_prefix=Arg("--image-prefix", "-i", help="Base url for page image URLs for "),
)
def annotate_poetry_with_image(
    dataset: str, source: str, labels: str, image_prefix: str = None
):
    nlp = spacy.blank("en")  # use blank spaCy model for tokenization
    stream = JSONL(source)  # load jsonlines into stream

    # ensure image prefix does not have a trailing slash
    if image_prefix is None:
        image_prefix = ""
    image_prefix = image_prefix.rstrip("/")

    def tokenize_stream(stream):
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
            # point to image server
            if "image_path" in task:
                task["image"] = f"{image_prefix}/{task['image_path']}"
            yield task

    tokenized_stream = tokenize_stream(stream)

    # split labels by commas and strip any whitespace
    label_list = [label.strip() for label in labels.split(",")]

    blocks = [
        {
            "view_id": "image_manual",
            "image_manual_spans_key": "image_spans",
            "labels": label_list,
        },
        {"view_id": "spans_manual", "labels": label_list},
    ]

    return {
        "dataset": dataset,
        "stream": tokenized_stream,
        "view_id": "blocks",
        "config": {
            "blocks": blocks,
            "labels": label_list,
            "show_flag": True,  # show flag button to mark weird/difficult examples
            "hide_newlines": False,  # ensure newlines are shown \n
            "allow_newline_highlight": True,  # allow highlighting of newlines \n
            "honor_token_whitespace": True,  # reflect whitespace accurately (e.g. in case of leading/trailing spaces)
            "custom_theme": {
                "labels": {
                    # trying to use PPA colors but doesn't quite work; can we customize highlight color?
                    "POETRY": "#f05b69",  # label color for POETRY
                    # "PROSODY": "#4661ac"  # label color for PROSODY
                },
                "hide_true_newline_tokens": False,
            },
            "global_css_dir": CURRENT_DIR,
        },
    }


# save this script as annotate_poetry_with_image.py
# then in the prodigy env, on the command line, run:

# export PRODIGY_ALLOWED_SESSIONS=annotator1,annotator2,annotator3

## prodigy annotate_poetry_with_image poetry_and_image_dataset data/testset-db.jsonl "POETRY,PROSODY" -F annotate_poetry_with_image.py

# then, to get the annotations out, run:

## prodigy db-out poetry_and_image_dataset > annotations.jsonl
