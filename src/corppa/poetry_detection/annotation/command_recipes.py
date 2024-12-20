"""
This module contains custom command recipes for Prodigy.

Recipes:
* `ppa-task-progress`: Report the current progress for a PPA annotation task
  at the page and annotator level.


Example use:
```
prodigy ppa-stats task_id -F command_recipes.py
```
"""

from collections import Counter, defaultdict

from prodigy.components.db import connect
from prodigy.core import Arg, recipe
from prodigy.errors import RecipeError
from prodigy.util import SESSION_ID_ATTR, msg


@recipe(
    "ppa-task-progress",
    dataset=Arg(help="Prodigy dataset ID"),
)
def page_stats(dataset: str) -> None:
    """
    This command reports the current progress for a PPA annotation task
    (specified by its dataset ID) at the page and annotator level. This
    command prints two tables:
        1. Reports the number of pages with k annotations
        2. Reports the number of annotations per session (i.e. annotator)
    """
    # Load examples from database
    DB = connect()
    if dataset not in DB:
        raise RecipeError(f"Can't find dataset '{dataset}' in database")
    examples = DB.get_dataset_examples(dataset)
    n_examples = len(examples)
    msg.good(f"Loaded {n_examples} annotations from {dataset} dataset")

    # Get page and session level statistics
    ## Tracks the number of examples for each page
    examples_by_page = Counter()
    ## Tracks the number of examples for each session
    examples_by_session = defaultdict(list)
    for ex in examples:
        # Skip examples without answer or (page) id
        if "answer" not in ex and "id" not in ex:
            # Ignore "unanswered" examples
            continue
        page_id = ex["id"]
        examples_by_page[page_id] += 1
        session_id = ex[SESSION_ID_ATTR]
        examples_by_session[session_id].append(page_id)
    ## Tracks the number of pages with k annotations
    count_freqs = Counter(examples_by_page.values())
    total = sum(examples_by_page.values())

    # Build overall table
    header = ["# Annotations"]
    row = ["# Pages"]
    for key, val in sorted(count_freqs.items()):
        header.append(f"{key}")
        row.append(val)
    header.append("Total")
    row.append(total)
    aligns = ["r", "r", "r", "r"]
    msg.table(
        [row],
        title="Overall Annotation Progress",
        header=header,
        aligns=aligns,
        divider=True,
    )

    # Build session table
    data = []
    cumulative_total = 0
    ## Sort sessions by annotation count in decreasing order
    for session, pages in sorted(
        examples_by_session.items(), key=lambda x: len(x[1]), reverse=True
    ):
        count = len(pages)
        unique = len(set(pages))
        cumulative_total += count
        row = [session, count, unique, cumulative_total]
        data.append(row)
    header = [
        "Session",
        "Count",
        "Unique",
        "Total",
    ]
    aligns = ["l", "r", "r", "r"]
    """
    If we'd like to have a legend for this table, we can replce the subseqeuent
    code with the following:
    
    # info = {
    #    "Session": "Session name",
    #    "Count": "Completed annotations",
    #    "Unique": "Unique annotations (distinct pages)",
    #    "Total": "Total annotations collected",
    # }
    # msg.table(info, title="Legend")
    """
    msg.table(
        data,
        title="Session Annotation Progress",
        header=header,
        aligns=aligns,
        divider=True,
    )
