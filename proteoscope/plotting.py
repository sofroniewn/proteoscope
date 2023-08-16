import numpy as np
import seaborn_image as isns
from ipywidgets import interact


def contrast_normalize(image, percentiles):
    vals = np.percentile(image, percentiles, keepdims=True)
    return (image - vals[0]) / (vals[1] - vals[0])


def merge_prot_nuc(image, percentiles=None):
    img_A = image[:, 0]
    img_B = image[:, 1]

    if percentiles is not None:
        img_A = contrast_normalize(img_A, percentiles)
        img_B = contrast_normalize(img_B, percentiles)

    rgb = np.tile(img_A[:, np.newaxis, :, :], (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    rgb[:, :, :, 2] = np.maximum(img_B, img_A)
    return np.clip(rgb, 0, 1)


def browse_reconstructions(A, B, true_names):
    unique_names = np.unique(true_names)

    def view_image(name):
        keep = true_names == name
        true = A[keep]
        predicted = B[keep]
        n = min(16, keep.sum())
        use = np.random.choice(keep.sum(), size=n, replace=False)
        isns.ImageGrid(
            true[use], col_wrap=8, cbar=False, height=1.5, axis=0, cmap="viridis"
        )
        isns.ImageGrid(
            predicted[use], col_wrap=8, cbar=False, height=1.5, axis=0, cmap="viridis"
        )

    interact(view_image, name=unique_names)
