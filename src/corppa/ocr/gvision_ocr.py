#!/usr/bin/env python

"""
setup:
 - in google cloud console, enable vision api for desired project
 - install gcloud cli (sdk) and login (this is now preferred to service accounts)
 - install python client: `pip install google-cloud-vision`
"""

import os
import sys
import io
import pathlib
import argparse

from tqdm import tqdm

# Attempt to import Google Cloud Vision Python Client
try:
    from google.cloud import vision
except ImportError:
    print(
        "Error: Python environment does not contain google-cloud-vision "
        "package. Switch environments or install package and try again."
    )
    sys.exit(1)


def image_relpath_generator(image_dir, ext_set, follow_symlinks=True):
    for dirpath, dirs, files in image_dir.walk(follow_symlinks=follow_symlinks):
        # Check the files in walked directory
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext:
                filepath = dirpath.joinpath(file)
                yield filepath.relative_to(image_dir)
        # For future walking, remove hidden directories
        dirs[:] = [d for d in dirs if d[0] != "."]


def get_image_ocr(in_dir, out_dir, ext_list, ocr_limit=0, show_progress=True):
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
    for image_relpath in image_relpath_generator(in_dir, set(ext_list)):
        # Refresh progress bar
        if show_progress:
            progress_bar.refresh()
        imagefile = in_dir.joinpath(image_relpath)
        textfile = out_dir.joinpath(image_relpath).with_suffix(".txt")
        jsonfile = out_dir.joinpath(image_relpath).with_suffix(".json")

        # Request OCR if file does not exist
        if not textfile.is_file():
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

            # Save json responst
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
                progress_bar.update()

            # Check if we should stop
            if ocr_limit and ocr_count == ocr_limit:
                # TODO: Is there a better structuring to avoid this break
                break

    if show_progress:
        # Close progress bar
        progress_bar.close()


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
        help="Accepted file extension(s). Defaults: .TIF, .jpg",
        nargs="*",
        type=str,
        action="extend",
    )

    args = parser.parse_args()
    # Workaround: Set default extensions if none are provided.
    if args.ext is None:
        args.ext = [".TIF", ".jpg"]

    # Validate arguments
    if not args.input.is_dir():
        print(f"Error: input directory {args.input} does not exist")
        sys.exit(1)
    # TODO: Is this too restrictive / unnecessary?
    if not args.output.is_dir():
        print(f"Error: output directory {args.output} does not exist")
        sys.exit(1)
    if args.ocr_limit < 0:
        print("Error: ocr limit cannot be negative")
        sys.exit(1)

    # TODO: Add try block?
    get_image_ocr(
        args.input,
        args.output,
        set(args.ext),
        ocr_limit=args.ocr_limit,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
