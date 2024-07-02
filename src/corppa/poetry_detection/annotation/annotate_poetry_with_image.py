import prodigy
from prodigy.components.loaders import JSONL
import spacy


@prodigy.recipe("annotate_poetry_with_image")
def annotate_poetry_with_image(dataset: str, source: str, labels: str):
    nlp = spacy.blank("en")  # use blank spaCy model for tokenization
    stream = JSONL(source)  # load jsonlines into stream

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
                task["image"] = f"http://localhost:8000/{task['image_path']}"
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
                    "POETRY": "#FFA500",  # label color for POETRY
                    # "PROSODY": "#00BFFF"  # label color for PROSODY
                },
                "hide_true_newline_tokens": False,
            },
        },
    }


# save this script as annotate_poetry_with_image.py
# then in the prodigy env, on the command line, run:

# export PRODIGY_ALLOWED_SESSIONS=annotator1,annotator2,annotator3

## prodigy annotate_poetry_with_image poetry_and_image_dataset data/testset-db.jsonl "POETRY,PROSODY" -F annotate_poetry_with_image.py

# then, to get the annotations out, run:

## prodigy db-out poetry_and_image_dataset > annotations.jsonl
