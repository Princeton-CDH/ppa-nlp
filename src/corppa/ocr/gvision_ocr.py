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

from corppa.utils.path_utils import get_ppa_source, get_vol_dir

# Attempt to import Google Cloud Vision Python Client
try:
    from google.cloud import vision
except ImportError:
    vision = None

# Workaround (hopefully temporary) to surpress some logging printed to stderr
os.environ["GRPC_VERBOSITY"] = "NONE"


def image_relpath_generator(image_dir, ext_set, follow_symlinks=True):
    """
    This generator method finds all images in image_dir with file extensions
    in ext_set. For each of these images, the method yields the relative path
    with respect to image_dir.

    For example, if image_dir = "a/b/c/images" and there are image files at the
    following paths: "a/b/c/images/alpha.jpg", "a/b/c/images/d/beta.jpg"
    The generate will produce these two items: "alpha.jpg" and "d/beta.jpg"
    """
    # Using pathlib.walk over glob because (1) it allows us to find files with
    # multiple extensions in a single walk of the directory and (2) lets us
    # leverage additional functionality of pathlib.
    for dirpath, dirs, files in image_dir.walk(follow_symlinks=follow_symlinks):
        # Check the files in walked directory
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext:
                filepath = dirpath.joinpath(file)
                yield filepath.relative_to(image_dir)
        # For future walking, remove hidden directories
        dirs[:] = [d for d in dirs if d[0] != "."]


def ocr_images(in_dir, out_dir, ext_list, ocr_limit=0, show_progress=True):
    """
    OCR images in in_dir with extension ext_list to out_dir. If ocr_limit > 0,
    stop after OCRing ocr_limit images.

    Returns a map structure reporting the number of images OCR'd and skipped.
    """
    # Instantiate google vision client
    client = vision.ImageAnnotatorClient()

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
    for image_relpath in image_relpath_generator(in_dir, set(ext_list)):
        # Refresh progress bar
        if show_progress:
            progress_bar.refresh()
        # Get image and ocr output paths
        imagefile = in_dir.joinpath(image_relpath)
        textfile = out_dir.joinpath(image_relpath).with_suffix(".txt")
        jsonfile = textfile.with_suffix(".json")
        # Ensure that all subdirectories exist
        ocr_dir = textfile.parent
        ocr_dir.mkdir(parents=True, exist_ok=True)

        # Request OCR if file does not exist
        if textfile.is_file():
            skip_count += 1
        else:
            try:
                # Load the image into memory
                with io.open(imagefile, "rb") as image_reader:
                    content = image_reader.read()
                image = vision.Image(content=content)

                # Performs OCR and handwriting detection on the image file
                response = client.document_text_detection(image=image)

                # Save plain text output to local file;
                # even if text is empty, create text file so we don't request again
                with open(textfile, "w") as textfilehandle:
                    textfilehandle.write(response.full_text_annotation.text)

                # Save json response
                json_response = vision.AnnotateImageResponse.to_json(response)
                with open(jsonfile, "w") as jsonfilehandle:
                    jsonfilehandle.write(json_response)

                if response.error.message:
                    raise Exception(
                        f"{response.error.message}\n for more info on error messages, "
                        "check: https://cloud.google.com/apis/design/errors"
                    )

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


def ocr_volumes(vol_ids, in_dir, out_dir, ext_list, ocr_limit=0, show_progress=True):
    """
    OCR images for volumes vol_ids with extension ext_list to out_dir. Assumes in_dir
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
            ext_list,
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
        help="Accepted file extension(s). Can be repeated. Defaults: .TIF, .jpg",
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
        args.ext = [".TIF", ".jpg"]

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
            set(args.ext),
            ocr_limit=args.ocr_limit,
            show_progress=args.progress,
        )
    else:
        ocr_volumes(
            args.vol,
            args.input,
            args.output,
            set(args.ext),
            ocr_limit=args.ocr_limit,
            show_progress=args.progress,
        )


if __name__ == "__main__":
    # Check that Google Cloud Vision Python Client was successfully imported
    if vision is None:
        print(
            "Error: Python environment does not contain google-cloud-vision "
            "package. Switch environments or install package and try again."
        )
        sys.exit(1)

    main()
