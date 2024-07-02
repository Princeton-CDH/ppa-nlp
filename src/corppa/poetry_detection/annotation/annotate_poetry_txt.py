import prodigy
from prodigy.components.loaders import JSONL
import spacy


@prodigy.recipe("ppa_poetry_annotation")
def ppa_poetry_annotation(dataset: str, source: str, labels: str):
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
                task["image_path"] = f"http://localhost:8000/{task['image_path']}"
            yield task

    tokenized_stream = tokenize_stream(stream)

    # split labels by commas and strip any whitespace
    label_list = [label.strip() for label in labels.split(",")]

    blocks = [
        {"view_id": "html", "html_template": "<img src='{{image_path}}' width='500'>"},
        {"view_id": "spans_manual", "labels": label_list},
        # {"view_id": "text", "labels": label_list},
    ]

    return {
        "dataset": dataset,
        "stream": tokenized_stream,
        "view_id": "blocks",
        "config": {
            "blocks": blocks,
            "labels": label_list,
            "show_flag": True,  # show flag button to mark weird/difficult examples
            "hide_newlines": False,  # ensure newlines are shown
            "allow_newline_highlight": True,  # allow highlighting newlines
            "honor_token_whitespace": True,  # reflect whitespace accurately
            "custom_theme": {
                "labels": {
                    "POETRY": "#FFA500",  # label color for POETRY
                    "PROSODY": "#00BFFF",  # label color for PROSODY
                },
                "hide_true_newline_tokens": False,
            },
        },
    }


# save this script as annotate_poetry_txt.py
# then in the prodigy env, on the command line, run:

# export PRODIGY_ALLOWED_SESSIONS=annotator1,annotator2,annotator3

## prodigy ppa_poetry_annotation my_poetry_dataset data/testset-db.jsonl "POETRY,PROSODY" -F annotate_poetry_txt.py

# then, to get the annotations out, run:

## prodigy db-out my_poetry_dataset > annotations.jsonl
