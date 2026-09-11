import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # list of works with known alignment - must have zipfile present locally to run this
    works = [
        "hvd.hn1dlr",
        "hvd.32044048962955-p145",
        "hvd.32044050831999-p50",
        "hvd.32044092711431-p469",
        "mdp.39015030933512",
        "mdp.39015030934866",
        "mdp.39015031107785",
        "mdp.39015058693683",
        "mdp.39015059390321",
        "mdp.39015059409386",
        "mdp.39015059409410",
        "nyp.33433066585435",
        "yale.39002004065844-p76",
    ]

    select_work = mo.ui.multiselect(
        # preselect first one so we don't have to worry about it being unset
        options=works,
        max_selections=1,
        value=[works[0]],
    )
    select_work
    return (select_work,)


@app.cell
def _(select_work):
    from pathlib import Path

    import polars as pl

    from corppa.utils.dataset_prep import (
        get_ht_zipfile_path,
        plot_alignment,
        review_alignment,
    )

    # work_id = "hvd.hn1dlr"
    work_id = select_work.value[0]
    # work_id = "hvd.32044048962955-p145"
    # work_id = "hvd.32044050831999-p50"
    # work_id = "hvd.32044092711431-p469"

    data_dir = Path("/Users/rkoeser/workarea/experiments/ppa-textimage-alignment")

    all_pages_path = Path(
        "~/workarea/experiments/ppa-textimage-alignment/ppa_corpus_2026-01-07_091133/ppa_pages.jsonl"
    ).expanduser()
    print(all_pages_path)
    # load original pages
    orig_pages_df = (
        pl.scan_ndjson(all_pages_path).filter(work_id=work_id).collect().sort("order")
    )
    orig_pages_df

    zip_path = get_ht_zipfile_path(work_id, data_dir)
    df = review_alignment(
        work_id, orig_pages_df, zip_path
    )  # pages: list[dict] or DataFrame
    plot_alignment(df)
    return df, pl


@app.cell
def _(df, pl):
    df.filter(~pl.col.is_matched)  # inspect gaps
    return


if __name__ == "__main__":
    app.run()
