"""
Generate image-level metadata using source-level metadata and page images.

Environment: ppa-images
"""

import os
import sys
import re
import csv
import argparse

from htrc_features.utils import clean_htid


def is_gale(source_id):
    """
    Check if source id corresponds to Gale or HathiTrust
    Returns: bool
    """
    return source_id[0].isupper()


def clean_source_id(source_id):
    """
    Clean the source_id, this will only impact certain HathiTrust sources.
    Returns: str
    """
    if is_gale(source_id):
        # Skip Gale sources
        return source_id
    else:
        # Clean HathiTrust sources
        return clean_htid(source_id)


def extract_page_no(source_id, fname):
    """
    Extract the page number from the filname fname for a page
    from the source source_id.
    Returns: int
    """
    # Set regex pattern specific to catalog
    if is_gale(source_id):
        # Gale page
        re_pattern = r"Page_(\d+)_"
    else:
        # HathiTrust page
        re_pattern = r"(\d+).jpg"
    match = re.search(re_pattern, fname)
    return int(match.group(1))


def load_metadata(csv_file, key_field):
    """ "
    Load metadata from csv_file using key_field as the output dict's keys
    Returns dict: key_field --> dict
    """
    meta = {}
    with open(csv_file, newline="") as file_handler:
        reader = csv.DictReader(file_handler, dialect="excel-tab")
        for row in reader:
            meta[row[key_field]] = row
    return meta


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Preprocess image data.")
    parser.add_argument("source-meta", help="Input source-level metadata file (tsv)")
    parser.add_argument("images", help="Top-level directory containing images")
    parser.add_argument("image-meta", help="Output image-level metdata file (tsv)")
    args = vars(parser.parse_args())

    source_tsv = args["source-meta"]
    image_dir = args["images"]
    image_tsv = args["image-meta"]

    # Validate input arguments
    if not os.path.isfile(source_tsv):
        print(f"ERROR: file '{source_tsv}' does not exist")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"ERROR: directory '{image_dir}' does not exist")
        sys.exit(1)

    source_meta = load_metadata(source_tsv, "source_id")

    image_fields = [
        "image_name",
        "index",
        "filename",
        "source_id",
        "page_no",
        "year",
        "source_title",
        "author",
    ]

    with open(image_tsv, mode="w", newline="") as file_handler:
        writer = csv.DictWriter(
            file_handler,
            dialect="excel-tab",
            fieldnames=image_fields,
        )
        writer.writeheader()

        i = 0
        for source_id in source_meta:
            source = clean_source_id(source_id)
            source_dir = os.path.join(image_dir, source)

            for f in os.listdir(source_dir):
                # Skip non-jpg files
                if not f.endswith(".jpg"):
                    continue
                fname = os.path.join(source, f)
                page_no = extract_page_no(source_id, fname)

                # Create metadata entry
                entry = {
                    "image_name": f"{source_id}-{page_no}",
                    "index": i,
                    "filename": fname,
                    "source_id": source_id,
                    "page_no": page_no,
                    "year": source_meta[source_id]["pub_date"],
                    "source_title": source_meta[source_id]["title"],
                    "author": source_meta[source_id]["author"],
                }
                # Write entry to file
                writer.writerow(entry)

                # Increment index
                i += 1


if __name__ == "__main__":
    main()
