#!/usr/bin/env python

"""
This script OCRs images using the Google Vision API.
"""

import argparse
import io
import os
import pathlib
import sys

from tqdm import tqdm

from corppa.utils.path_utils import find_relative_paths, get_ppa_source, get_vol_dir

# Attempt to import Google Cloud Vision Python Client
try:
    from google.cloud import vision as google_vision
except ImportError:
    google_vision = None

# Workaround (hopefully temporary) to surpress some logging printed to stderr
os.environ["GRPC_VERBOSITY"] = "NONE"


def ocr_image_via_gvision(gvision_client, input_image, out_txt, out_json):
    """
    Perform OCR for input image using the Google Cloud Vision API via the provided client.
    The plaintext output and json response of the OCR call are written to out_txt and
    out_json paths respectively.
    """
    # TODO: Clean up code duplication. This check is needed, since this method relies on
    #       both an existing client as well as API calls directly.
    # Check that Google Cloud Vision Python Client library was successfully imported
    if google_vision is None:
        print(
            "Error: Python environment does not contain google-cloud-vision "
            "package. Switch environments or install package and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load the image into memory
    with io.open(input_image, "rb") as image_reader:
        content = image_reader.read()
        image = google_vision.Image(content=content)

        # Performs OCR and handwriting detection on the image file
        response = gvision_client.document_text_detection(image=image)

        # Save plain text output to local file;
        # even if text is empty, create text file so we don't request again
        with open(out_txt, "w") as textfilehandle:
            textfilehandle.write(response.full_text_annotation.text)

        # Save json response
        json_response = google_vision.AnnotateImageResponse.to_json(response)
        with open(out_json, "w") as jsonfilehandle:
            jsonfilehandle.write(json_response)

        if response.error.message:
            raise Exception(
                f"{response.error.message}\n for more info on error messages, "
                "check: https://cloud.google.com/apis/design/errors"
            )


def ocr_images(in_dir, out_dir, exts, ocr_limit=0, show_progress=True):
    """
    OCR images in in_dir with extension exts to out_dir. If ocr_limit > 0,
    stop after OCRing ocr_limit images.

    Returns a map structure reporting the number of images OCR'd and skipped.
    """
    # Check that Google Cloud Vision Python Client was successfully imported
    if google_vision is None:
        print(
            "Error: Python environment does not contain google-cloud-vision "
            "package. Switch environments or install package and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Instantiate google vision client
    client = google_vision.ImageAnnotatorClient()

    # Setup up progress bar if progress will be shown
    if show_progress:
        desc = "OCRing images"
        maxinterval = 1
        if ocr_limit:
            progress_bar = tqdm(desc=desc, total=ocr_limit, maxinterval=maxinterval)
        else:
            bar_format = "{desc}: {n:,} images OCR'd | elapsed: {elapsed}, {rate_fmt}"
            progress_bar = tqdm(
                desc=desc, bar_format=bar_format, maxinterval=maxinterval
            )

    ocr_count = 0
    skip_count = 0
    for image_relpath in find_relative_paths(in_dir, exts):
        # Refresh progress bar
        if show_progress:
            progress_bar.refresh()
        # Get image and ocr output paths
        image_file = in_dir.joinpath(image_relpath)
        text_file = out_dir.joinpath(image_relpath).with_suffix(".txt")
        json_file = text_file.with_suffix(".json")
        # Ensure that all subdirectories exist
        ocr_dir = text_file.parent
        ocr_dir.mkdir(parents=True, exist_ok=True)

        # Request OCR if file does not exist
        if text_file.is_file():
            skip_count += 1
        else:
            try:
                ocr_image_via_gvision(client, image_file, text_file, json_file)

                # Update counter
                ocr_count += 1
                if show_progress:
                    # Update progress bar since only OCR'd images are tracked
                    progress_bar.update()

                # Check if we should stop
                if ocr_limit and ocr_count == ocr_limit:
                    # TODO: Is there a better structuring to avoid this break
                    break
            except (Exception, KeyboardInterrupt):
                # Close progress bar before raising error
                progress_bar.close()
                print(
                    f"Error: An error encountered while OCRing {imagefile.stem}",
                    file=sys.stderr,
                )
                raise

    if show_progress:
        # Close progress bar
        progress_bar.close()
        if ocr_limit and ocr_count == ocr_limit:
            print("Stopping early, OCR limit reached.", file=sys.stderr)
        print(
            f"{ocr_count:,} images OCR'd & {skip_count:,} images skipped.",
            file=sys.stderr,
        )

    return {"ocr_count": ocr_count, "skip_count": skip_count}


def ocr_volumes(vol_ids, in_dir, out_dir, exts, ocr_limit=0, show_progress=True):
    """
    OCR images for volumes vol_ids with extension exts to out_dir. Assumes in_dir
    follows the PPA directory conventions (see corppa.utils.path_utils for more
    details). If ocr_limit > 0, stop after OCRing ocr_limit images.
    """
    n_vols = len(vol_ids)
    current_ocr_limit = ocr_limit
    total_ocr_count = 0
    total_skip_count = 0
    for i, vol_id in enumerate(vol_ids):
        try:
            sub_dir = get_vol_dir(vol_id)
        except NotImplementedError:
            # Skip unsupported source types (i.e. HathiTrust)
            vol_source = get_ppa_source(vol_id)
            print(
                f"Warning: Skipping {vol_id} since its source ({vol_source}) is "
                "not yet unsupported.",
                file=sys.stderr,
            )
            continue

        # Get vol dir info
        in_vol_dir = in_dir.joinpath(sub_dir)
        out_vol_dir = out_dir.joinpath(sub_dir)

        # Check that input vol dir exists
        if not in_vol_dir.is_dir():
            print(f"Warning: Volume '{vol_id}' is not in {in_dir}", file=sys.stderr)
            print(f"Directory {in_vol_dir} does not exist.", file=sys.stderr)
            continue
        # Ensure that output vol dir exists
        out_vol_dir.mkdir(parents=True, exist_ok=True)
        if show_progress:
            # Add space between volume-level reporting
            if i:
                print("", file=sys.stderr)
            print(f"OCRing {vol_id} ({i+1}/{n_vols})...", file=sys.stderr)

        # OCR images
        report = ocr_images(
            in_vol_dir,
            out_vol_dir,
            exts,
            ocr_limit=current_ocr_limit,
            show_progress=show_progress,
        )

        # Upkeep
        total_ocr_count += report["ocr_count"]
        total_skip_count += report["skip_count"]
        if ocr_limit:
            current_ocr_limit -= report["ocr_count"]
            # Stop if limit is reached
            if current_ocr_limit == 0:
                if show_progress:
                    print("Hit OCR limit.", file=sys.stderr)
                break

    print(
        f"---\nIn total, {total_ocr_count:,} images OCR'd & {total_skip_count:,} "
        "images skipped.",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Uses Google Vision API to OCR images."
    )

    # Required arguments
    parser.add_argument(
        "input",
        help="Top-level input directory containing images to be OCR'd",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Top-level output directory for OCR output; "
        + "maintains input subdirectory structure.",
        type=pathlib.Path,
    )

    # Optional arguments
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ocr-limit",
        help="Set a limit for the number of images to be OCR'd",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ext",
        help="Accepted file extension(s), case insensitive. Can be repeated. Defaults: .tif, .jpg",
        nargs="*",
        type=str,
        action="extend",
    )
    parser.add_argument(
        "--vol",
        help="Only OCR images from the specified PPA volume(s) represented as "
        "volume ids. Can be repeated.",
        nargs="*",
        action="extend",
    )

    args = parser.parse_args()
    # Workaround: Set default extensions if none are provided.
    if args.ext is None:
        args.ext = [".tif", ".jpg"]

    # Validate arguments
    if not args.input.is_dir():
        print(f"Error: input directory {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    # TODO: Is this too restrictive / unnecessary?
    if not args.output.is_dir():
        print(f"Error: output directory {args.output} does not exist", file=sys.stderr)
        sys.exit(1)
    if args.ocr_limit < 0:
        print("Error: ocr limit cannot be negative", file=sys.stderr)
        sys.exit(1)

    if args.vol is None:
        ocr_images(
            args.input,
            args.output,
            args.ext,
            ocr_limit=args.ocr_limit,
            show_progress=args.progress,
        )
    else:
        ocr_volumes(
            args.vol,
            args.input,
            args.output,
            args.ext,
            ocr_limit=args.ocr_limit,
            show_progress=args.progress,
        )


if __name__ == "__main__":
    main()
