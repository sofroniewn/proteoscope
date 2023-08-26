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


def browse_reconstructions(A, B, names=None, n_im=16):
    if names is not None:
        unique_names = np.unique(names)
    else:
        unique_names = None

    def view_image(name=None):
        if name is not None:
            keep = names == name
            A_ = A[keep]
            B_ = B[keep]
        else:
            A_ = A
            B_ = B

        n = min(n_im, A_.shape[0])
        use = np.random.choice(A_.shape[0], size=n, replace=False)
        isns.ImageGrid(
            A_[use], col_wrap=n_im // 2, cbar=False, height=1.5, axis=0, cmap="viridis"
        )
        isns.ImageGrid(
            B_[use], col_wrap=n_im // 2, cbar=False, height=1.5, axis=0, cmap="viridis"
        )

    if unique_names is not None:
        interact(view_image, name=unique_names)
    else:
        view_image()
