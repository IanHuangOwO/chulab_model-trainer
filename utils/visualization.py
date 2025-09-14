"""
Simple visualization utilities for image and mask pairs.

- Training datasets (image, mask): two columns per sample
  left = image (grayscale), right = mask (grayscale)
- Inference datasets (image only): one column per sample
"""

import numpy as np
import torch
from typing import Optional

def _to_numpy(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return arr[0]
    return arr

def visualize_dataset(dataset, rows: int = 5, pairs_per_row: int = 4, title: Optional[str] = None) -> None:
    """
    Display a fixed grid of 9 rows x 16 columns (8 pairs per row):
    - Each pair spans two columns: left=image, right=mask.
    - Samples with empty masks are skipped for mask datasets.
    - For image-only datasets, cells are filled with images sequentially.
    """
    import matplotlib.pyplot as plt

    total_pairs = pairs_per_row * rows
    n_items = len(dataset)
    if n_items == 0:
        return

    sample = dataset[0]
    has_mask = isinstance(sample, (tuple, list)) and len(sample) >= 2

    # Select indices
    if has_mask:
        selected: list[int] = []
        for idx in range(n_items):
            try:
                _, m = dataset[idx]
            except Exception:
                continue
            m = _squeeze_channel(_to_numpy(m))
            if np.any(m > 0):
                selected.append(idx)
                if len(selected) >= total_pairs:
                    break
        if not selected:
            return
    else:
        selected = list(range(min(total_pairs, n_items)))

    # Create figure: rows x (2*pairs_per_row) axes
    cols = 2 * pairs_per_row
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    if rows == 1:
        axes = np.array([axes])

    # Fill grid
    for k, idx_ds in enumerate(selected):
        r = k // pairs_per_row
        c_pair = k % pairs_per_row
        if r >= rows:
            break

        if has_mask:
            img_t, msk_t = dataset[idx_ds]
            img = _squeeze_channel(_to_numpy(img_t))
            msk = _squeeze_channel(_to_numpy(msk_t))
            # Choose z slice with most mask
            if img.ndim == 3 and msk.ndim == 3:
                sums = msk.sum(axis=(1, 2))
                z = int(np.argmax(sums))
                img2d, msk2d = img[z], msk[z]
            else:
                img2d = img[img.shape[0] // 2] if img.ndim == 3 else img
                msk2d = msk[msk.shape[0] // 2] if msk.ndim == 3 else msk

            ax_img = axes[r, 2 * c_pair]
            ax_msk = axes[r, 2 * c_pair + 1]
            ax_img.imshow(img2d, cmap="gray")
            ax_img.set_title(f"img[{idx_ds}]")
            ax_img.axis("off")
            ax_msk.imshow(msk2d, cmap="gray", vmin=0, vmax=1)
            ax_msk.set_title(f"msk[{idx_ds}]")
            ax_msk.axis("off")
        else:
            img_t = dataset[idx_ds]
            img = _squeeze_channel(_to_numpy(img_t))
            img2d = img[img.shape[0] // 2] if img.ndim == 3 else img
            ax = axes[r, c_pair]
            ax.imshow(img2d, cmap="gray")
            ax.set_title(f"img[{idx_ds}]")
            ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # Ignore save_path; display instead
    plt.show()
    plt.close(fig)
