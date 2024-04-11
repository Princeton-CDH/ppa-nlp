"""
Create the full, flattened embeddings (dinov2-small) for each image, and
optionally save preprocessed images.

Environment: ppa-images
"""

import sys
import csv
import os.path
import numpy as np
import argparse
import torchvision

from tqdm import tqdm
from transformers import pipeline, AutoImageProcessor
from PIL import Image


def get_image_processor(tmt):
    """
    Returns the image process for a given treatment tmt
    """
    if tmt == "default":
        # Use dinov2-small's default settings;
        return AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    elif tmt == "resize-crop":
        return AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small", size={"shortest_edge": 224}
        )
    elif tmt == "resize-only":
        return AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small", size={"height": 224, "width": 224}
        )
    else:
        print(f"ERROR: unsupported treatment '{tmt}'")
        sys.exit(1)


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Extract image embeddings.")
    parser.add_argument("image-meta", help="Input image-level metadata file (tsv)")
    parser.add_argument("input-dir", help="Top-level directory containing input images")
    parser.add_argument("output-embeddings", help="Output embeddings file (npy)")
    parser.add_argument("--tmt", default="default", help="Preprocessing treatment")
    parser.add_argument(
        "--save-images",
        help="Save preprocessed images in the specified " "top-level directory",
    )

    args = vars(parser.parse_args())

    meta_tsv = args["image-meta"]
    images_dir = args["input-dir"]
    out_npy = args["output-embeddings"]
    tmt = args["tmt"]
    out_dir = args.get("save_images")

    # Validate input arguments
    if not os.path.isfile(meta_tsv):
        print(f"ERROR: file '{meta_tsv}' does not exist")
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print(f"ERROR: directory '{images_dir}' does not exist")
        sys.exit(1)
    if out_dir is not None and not os.path.isdir(out_dir):
        print(f"ERROR: directory '{out_dir}' does not exist")
        sys.exit(1)

    # Get images filenames (with subdirectory prefix)
    image_fnames = []
    with open(meta_tsv, newline="") as file_handler:
        reader = csv.DictReader(file_handler, dialect="excel-tab")
        for row in tqdm(reader):
            fname = row["filename"]
            image_fnames.append(fname)

    # Extract each image embedding, optionally saving the preprocessed image
    feature_set = None
    processor = get_image_processor(tmt)
    extractor = pipeline(
        model="facebook/dinov2-small",
        task="image-feature-extraction",
        image_processor=processor,
    )
    for image_fname in tqdm(image_fnames):
        image_path = os.path.join(images_dir, image_fname)

        # Optionally, preprocess & save image
        if out_dir:
            out_fpath = os.path.join(out_dir, image_fname)
            # Create any subdirectories if they doesn't exist
            full_dpath = os.path.dirname(out_fpath)
            if not os.path.isdir(full_dpath):
                os.makedirs(full_dpath)
            preprocessed_tensor = next(
                iter(
                    processor.preprocess(
                        Image.open(image_path), return_tensors="pt"
                    ).values()
                )
            )
            torchvision.utils.save_image(preprocessed_tensor, out_fpath)

        feats = extractor(image_path, return_tensors=True).cpu().numpy()
        # Flatten 2D features to 1D: (257, 384) --> 98688
        feats = feats.flatten()
        if feature_set is None:
            feature_set = feats
        else:
            feature_set = np.vstack((feature_set, feats))

    # Save extracted features
    np.save(out_npy, feature_set)


if __name__ == "__main__":
    main()
