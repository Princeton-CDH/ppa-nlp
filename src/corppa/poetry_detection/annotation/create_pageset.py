"""
Create poetry page set.
Note that this is hard-coded for the poetry test-set

env: ppa-data
"""

import csv
import os.path
import json
import re
import sys

import orjsonl
from xopen import xopen
from tqdm import tqdm
from helper import encode_htid, get_stub_dir


def extract_page_numbers(page_url_list):
    pg_urls = page_url_list.split("\n")
    pg_nums = {int(url.rsplit("=", 1)[1]) for url in pg_urls}
    return pg_nums


def get_page_image_path(page_record):
    source = page_record["source"]
    vol_id = page_record["source_id"]
    stub_dir = get_stub_dir(source, vol_id)
    page_num = page_record["order"]
    if source == "Gale":
        vol_dir = f"Gale/{stub_dir}/{vol_id}"
        image_name = f"{vol_id}_{page_num:04d}0.TIF"
        return f"{vol_dir}/{image_name}"
    elif source == "HathiTrust":
        vol_id = encode_htid(vol_id)
        ver_date = page_record["ver_date"]
        vol_dir = f"HathiTrust/{stub_dir}/{vol_id}_{ver_date}"
        image_name = f"{page_num:08d}.jpg"
        return f"{vol_dir}/{image_name}"
    else:
        print(f"ERROR: Unknown source '{source}'")
        raise ValueError


def get_ver_date(possible_timestamp):
    pattern = re.compile(r"\d\d\d\d-\d\d-\d\d")

    result = pattern.match(possible_timestamp.strip())
    if result:
        # Valid date identified
        return result.group()
    else:
        # No date found
        return "N/A"


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: [ppa-text corpus dir] [pageset csv] [out jsonl]")
        sys.exit(1)

    ppa_dir = sys.argv[1]
    pageset_csv = sys.argv[2]
    out_jsonl = sys.argv[3]

    ppa_meta_json = f"{ppa_dir}/ppa_metadata.json"
    ppa_jsonl = f"{ppa_dir}/ppa_pages.jsonl.gz"

    # Validate inputs
    if not os.path.isfile(pageset_csv):
        print(f"ERROR: {pageset_csv} does not exist")
        sys.exit(1)
    if not os.path.isdir(ppa_dir):
        print(f"ERROR: {ppa_dir} does not exist")
        sys.exit(1)
    if not os.path.isfile(ppa_meta_json):
        print("ERROR: PPA metadata file (ppa_metadata.json) does not exist")
        sys.exit(1)
    if not os.path.isfile(ppa_jsonl):
        print("ERROR: PPA pages file (ppa_pages.json.gz) does not exist")
        sys.exit(1)
    if os.path.isfile(out_jsonl):
        print(f"ERROR: {out_jsonl} already exists")
        sys.exit(1)

    # Load ppa metadata
    works_meta = {}
    with open(ppa_meta_json) as file_handler:
        for work in json.load(file_handler):
            works_meta[work["work_id"]] = work

    # Load testset data
    working_set = {}
    with open(pageset_csv, newline="") as file_handler:
        reader = csv.DictReader(file_handler)
        pg_rng_id = "digital page span of main text as determined by Mary"
        has_poetry_id = "links to pages with poetry (non-comprehensive)"
        for row in reader:
            work_id = row["ID"]
            work_record = works_meta[work_id]
            pg_start, pg_end = map(int, row[pg_rng_id].split("-"))
            entry = {
                "work_id": work_id,
                "source": work_record["source"],
                "source_id": work_record["source_id"],
                "source_url": work_record["source_url"],
                "pub_year": work_record["pub_year"],
                "pg_start": pg_start,
                "pg_end": pg_end,
                "poetry_pages": extract_page_numbers(row[has_poetry_id]),
                "ver_date": get_ver_date(row["version_date"]),
            }
            working_set[work_id] = entry

    # Gather pages
    n_lines = sum(1 for line in xopen(ppa_jsonl, mode="rb"))
    for page in tqdm(orjsonl.stream(ppa_jsonl), total=n_lines):
        work_id = page["work_id"]
        if work_id in working_set:
            work = working_set[work_id]
            page_num = page["order"]
            assert page_num == int(page["id"].rsplit(".", 1)[1])
            # Filter to working range of volume
            if page_num >= work["pg_start"] and page_num <= work["pg_end"]:
                # Add some additional metdata
                page["source"] = work["source"]
                page["source_id"] = work["source_id"]
                page["pub_year"] = work["pub_year"]
                page["ver_date"] = work["ver_date"]

                # Check if this page is known to contain poetry
                contains_poetry = "?"
                if page_num in work["poetry_pages"]:
                    contains_poetry = "Yes"
                page["contains_poetry"] = contains_poetry

                # Get image path (if possible)
                image_path = get_page_image_path(page)
                page["image_path"] = image_path

                # Write page data to file
                orjsonl.append(out_jsonl, page)
