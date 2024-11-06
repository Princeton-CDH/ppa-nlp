from collections import Counter, defaultdict

from prodigy.components.db import connect
from prodigy.core import Arg, recipe
from prodigy.errors import RecipeError
from prodigy.util import SESSION_ID_ATTR, msg


@recipe(
    "page-stats",
    dataset=Arg(help="Prodigy dataset ID"),
)
def ppa_stats(dataset: str) -> None:
    # Load examples
    DB = connect()
    if dataset not in DB:
        raise RecipeError(f"Can't find dataset '{dataset}' in database")
    examples = DB.get_dataset_examples(dataset)
    n_examples = len(examples)
    msg.good(f"Loaded {n_examples} annotations from {dataset} dataset")

    # Get stats
    examples_by_page = Counter()
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
    # Get frequencies of page-level annotation counts
    count_freqs = Counter()
    total = 0
    for count in examples_by_page.values():
        count_freqs[count] += 1
        total += count

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
    total = 0
    for session, pages in sorted(examples_by_session.items()):
        count = len(pages)
        unique = len(set(pages))
        total += count
        row = [session, count, unique, total]
        data.append(row)
    header = [
        "Session",
        "Count",
        "Unique",
        "Total",
    ]
    aligns = ["l", "r", "r", "r"]
    # info = {
    #    "Session": "Session name",
    #    "Count": "Completed annotations",
    #    "Unique": "Unique annotations (distinct pages)",
    #    "Total": "Total annotations collected",
    # }
    # msg.table(info, title="Legend")
    msg.table(
        data,
        title="Session Annotation Progress",
        header=header,
        aligns=aligns,
        divider=True,
    )
