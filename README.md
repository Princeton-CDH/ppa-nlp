# PPA - image projection

This branch contains experimental code for visualizing extracted feature
image embeddings of the pages within the PPA collection.


## Development Instructions
These instructions are for general development, as well as the construction
of all input needed for viewing the image projection.

### Setup conda environment
Create and activate the conda environment specified in `ppa_images.yml`.
```
conda env create -f ppa_images.yml
conda activate ppa-images
```

### Gather required data
The accompanying code relies on two key pieces of data.
1. Source-level Metadata: A tsv containing source-level metadata. Note that
there isn't currently support for excerpt-specific metadata from the same work.
2. Page Images: The page images. Currently, assumes that pages are organized
into directories by their source.

### Scripts
Here is the list of the existing python scripts located within
`src/image-projection`.

- `create_image_meta.py`: Create a page-level metadata file (tsv) from an
  initial source-level metadata file (tsv).
- `extract_features.py`: Extract the (full) feature embeddings for each page
   image. Optionally, save the preprocessed image form for each page.
- `reduce_features.py`: Reduce the dimensionality of the input embeddings (npy) and
  save the result in a format better suited for tensorboard (tsv).
- `create_sprite.py`: Create a sprite image (png) for the tensorboard.


## Viewing Image Projections
This must be run within the `tensorboard` directory.
```
cp template
```

### Setup and activate conda environment
Create and activate the conda environment specified in `tensorboard.yml`.
```
conda env create -f tensorboard.yml
conda activate tensorboard
```

Additionally make sure to move to the `tensorboard` directory
```
cd tensorboard
```

### Create and modify the projector config file
Copy the template config file `template.pbtxt` as `projector_config.pbtxt`.
Then update the various fields within the config.

### Running tensorboard
To launch tensorboard, run the following command.
```
tensorboard --logdir .
```

Then go visit [localhost:6006](https://localhost:6006) to launch tensorboard.
Then select the "Projector" tab from the drop-down menu on the top right. Note,
it should be the last item in the drop-down menu.
