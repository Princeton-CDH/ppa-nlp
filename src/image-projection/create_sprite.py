"""
Create sprite image for tensorboard embedding projector.
Note: This requires the tiles images to be square. Revisit later.

Environment: ppa-images
"""

import sys
import os.path
import csv
import math
import argparse

from tqdm import tqdm
from PIL import Image


# Limit set by tensorboard
MAX_SPRITE_IMAGE_SIZE = 8192


def main():
    parser = argparse.ArgumentParser(description="Create sprite for dataset.")
    parser.add_argument("image-meta", help="Input image-level metadata file (tsv)")
    parser.add_argument("image-dir", help="Top-level image directory")
    parser.add_argument("output-sprite", help="Output sprite image (png)")
    parser.add_argument(
        "--tile-size",
        default=50,
        type=int,
        help="Size (in pixels) for each image in sprite. " "Default: 50",
    )

    args = vars(parser.parse_args())

    image_meta = args["image-meta"]
    image_dir = args["image-dir"]
    out_sprite = args["output-sprite"]
    tile_size = args["tile_size"]

    # Validate input arguments
    if not os.path.isfile(image_meta):
        print(f"ERROR: file '{image_meta}' does not exist")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"ERROR: directory '{image_dir}' does not exist")
        sys.exit(1)
    if os.path.isfile(out_sprite):
        print(f"ERROR: file '{out_sprite}' already exists")
        sys.exit(1)

    # Collect image filepaths from image_meta
    image_fpaths = []
    with open(image_meta, newline="") as file_handler:
        reader = csv.DictReader(file_handler, dialect="excel-tab")
        for row in reader:
            fname = row["filename"]
            fpath = os.path.join(image_dir, fname)
            image_fpaths.append(fpath)

    # Build sprite file
    n_images = len(image_fpaths)
    grid = math.ceil(math.sqrt(n_images))

    # Determine working image size within sprite
    max_image_size = MAX_SPRITE_IMAGE_SIZE // grid
    image_size = min(tile_size, max_image_size)
    if image_size < tile_size:
        print(
            f"Error: Tile size {tile_size}px too large for the overall "
            f"sprite size limit. Select a tile size <= {max_image_size}px."
        )
        sys.exit(1)

    sprite_image = Image.new(
        mode="RGBA", size=(image_size * grid, image_size * grid), color=(0, 0, 0, 0)
    )  # fully transparent

    for i, fpath in enumerate(tqdm(image_fpaths)):
        row = i // grid
        col = i % grid
        image = Image.open(fpath)
        image = image.resize((image_size, image_size), Image.BICUBIC)
        row_loc = row * image_size
        col_loc = col * image_size

        # Note: row & col indices are reversed due to PIL saving
        sprite_image.paste(image, (col_loc, row_loc))
    sprite_image.save(out_sprite, "PNG")


if __name__ == "__main__":
    main()
