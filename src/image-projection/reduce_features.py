"""
Reduces the dimensionality of a set of feature embeddings.
Embeddings are saved into a format suitable for tensorboard (tsv).

Environment: ppa-images
"""

import sys
import os.path
import numpy as np
import argparse

from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser(
        description="Reduce the dimensionality of feature embeddings."
    )
    parser.add_argument("input-features", help="Input embeddings (npy)")
    parser.add_argument("dims", type=int, help="Size of output dimension")
    parser.add_argument("output-features", help="Output embeddings (tsv)")
    args = vars(parser.parse_args())

    in_npy = args["input-features"]
    dims = args["dims"]
    out_tsv = args["output-features"]

    if not os.path.isfile(in_npy):
        print(f"ERROR: file '{in_npy}' does not exist")
        sys.exit(1)
    if os.path.isfile(out_tsv):
        print(f"ERROR: file '{out_tsv}' already exists")
        sys.exit(1)

    feats = np.load(in_npy)
    # Reduce features to 'dims' dimensions
    print("Reducing features...")
    transformer = PCA(n_components=dims)
    reduced_feats = transformer.fit_transform(feats)
    explained_variance = sum(transformer.explained_variance_ratio_)
    print(f"Explained Variance: {explained_variance:.2%}")

    # Write features to file in tsv form
    print("Saving reduced features...")
    np.savetxt(out_tsv, reduced_feats, fmt="%.8f", delimiter="\t")


if __name__ == "__main__":
    main()
