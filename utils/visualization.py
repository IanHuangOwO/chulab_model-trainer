"""
Simple visualization utilities for image and mask pairs.

- Training datasets (image, mask): two columns per sample
  left = image (grayscale), right = mask (grayscale)
- Inference datasets (image only): one column per sample
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _to_numpy(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return arr[0]
    return arr

def visualize_dataset(dataset, n: int = 6, title: Optional[str] = None) -> None:
    # Show interactively using the current matplotlib backend; do not save to disk
    import matplotlib.pyplot as plt

    n = max(1, min(n, len(dataset)))
    sample = dataset[0]
    has_mask = isinstance(sample, (tuple, list)) and len(sample) >= 2

    cols = 2 if has_mask else 1
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(rows, 1)

    for i in range(n):
        if has_mask:
            img_t, msk_t = dataset[i]
            img = _squeeze_channel(_to_numpy(img_t))
            msk = _squeeze_channel(_to_numpy(msk_t))
            # pick slice where mask is most visible; only display mask if it has positives
            mask_has_any = bool(np.any(msk > 0))
            if img.ndim == 3:
                if mask_has_any and msk.ndim == 3:
                    sums = msk.sum(axis=(1, 2))
                    idx = int(np.argmax(sums))
                else:
                    idx = img.shape[0] // 2
                img2d = img[idx]
            else:
                img2d = img
            if mask_has_any:
                if msk.ndim == 3:
                    msk2d = msk[idx]
                else:
                    msk2d = msk
            else:
                msk2d = None
                
            axes[i, 0].imshow(img2d, cmap="gray")
            axes[i, 0].set_title(f"image[{i}]")
            axes[i, 0].axis("off")

            # Show mask only (no overlay) if present; else annotate
            if msk2d is not None and np.any(msk2d > 0):
                axes[i, 1].imshow(msk2d, cmap="gray", vmin=0, vmax=2)
                axes[i, 1].set_title(f"mask[{i}]")
                axes[i, 1].axis("off")
            else:
                axes[i, 1].axis("off")
                axes[i, 1].text(0.5, 0.5, "no mask", ha='center', va='center', fontsize=10)
        else:
            img_t = dataset[i]
            img = _squeeze_channel(_to_numpy(img_t))
            if img.ndim == 3:
                img2d = img[img.shape[0] // 2]
            else:
                img2d = img
            axes[i, 0].imshow(img2d, cmap="gray")
            axes[i, 0].set_title(f"image[{i}]")
            axes[i, 0].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # Ignore save_path; display instead
    plt.show()
    plt.close(fig)
