"""
Reduces the dimensionality of a set of feature embeddings.
Embeddings are saved into a format suitable for tensorboard (tsv).

Environment: ppa-images
"""

import sys
import os.path
import numpy as np
import argparse
import umap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_features(method, n_dims, features):
    """
    Reduce input numpy array features to n_dims dimensions using
    specified method
    Returns: numpy array
    """
    if method == "pca":
        return PCA(n_components=n_dims).fit_transform(features)
    elif method in ["tsne", "umap"]:
        # First, use PCA to reduce features to 50 dimensions
        if n_dims < 50:
            features = PCA(n_components=50).fit_transform(features)
        # Initialize reducer
        if method == "tsne":
            reducer = TSNE(n_components=n_dims)
        else:
            reducer = umap.UMAP(n_components=n_dims)
        return reducer.fit_transform(features)
    else:
        print(f"ERROR: Unsupported dimensionality reduction method '{method}'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Reduce the dimensionality of feature embeddings."
    )
    parser.add_argument("input-features", help="Input embeddings (npy)")
    parser.add_argument("output-features", help="Output embeddings (tsv)")
    parser.add_argument(
        "--dims", default=3, type=int, help="Size of output dimension (default: 3)"
    )
    parser.add_argument(
        "--method",
        default="pca",
        choices=["pca", "tsne", "umap"],
        help="Dimensionality reduction method (default: pca)",
    )
    args = vars(parser.parse_args())

    in_npy = args["input-features"]
    out_tsv = args["output-features"]
    dims = args["dims"]
    dr_method = args["method"]

    if not os.path.isfile(in_npy):
        print(f"ERROR: file '{in_npy}' does not exist")
        sys.exit(1)
    if os.path.isfile(out_tsv):
        print(f"ERROR: file '{out_tsv}' already exists")
        sys.exit(1)

    # Load in features
    feats = np.load(in_npy)

    print("Reducing features...")
    reduced_feats = reduce_features(dr_method, dims, feats)

    # Write features to file in tsv form
    print("Saving reduced features...")
    np.savetxt(out_tsv, reduced_feats, fmt="%.8f", delimiter="\t")


if __name__ == "__main__":
    main()
