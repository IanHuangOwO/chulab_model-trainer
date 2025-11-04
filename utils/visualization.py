"""
Simple visualization utilities for image and mask pairs.

- Handles training/validation datasets: (image, mask) tuples.
- Handles inference datasets: image-only tensors.
- Optionally displays predictions alongside ground truth.
- For 3D volumes, automatically selects the Z-slice with the most signal in the mask.
- Skips empty/background-only samples to show more relevant examples.
"""

import numpy as np
import torch
from typing import Optional, List

def _to_numpy(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    """Remove channel dimension if it's 1. (C, [D,] H, W) -> ([D,] H, W)."""
    if arr.ndim > 2 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _get_sample_data(dataset, idx: int, predictions: Optional[List[np.ndarray]] = None) -> tuple:
    """
    Safely extract image, mask, and prediction for a given index.
    Returns a tuple (image, mask, prediction) where mask and prediction can be None.
    """
    sample = dataset[idx]
    image, mask, pred = None, None, None

    if isinstance(sample, (tuple, list)):
        # Assumes (image, mask) format for training/validation datasets
        if len(sample) >= 2:
            image, mask = sample[0], sample[1]
        else:
            image = sample[0]
    else:
        # Assumes image-only format for inference datasets
        image = sample

    if predictions is not None and idx < len(predictions):
        pred = predictions[idx]

    return _to_numpy(image), _to_numpy(mask) if mask is not None else None, _to_numpy(pred) if pred is not None else None


def _select_best_z_slice(image: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple:
    """
    Select the best 2D slice from a 3D volume.
    - If a mask is provided, it picks the slice with the largest sum of mask values.
    - Otherwise, it picks the middle slice of the image.
    Returns (image_2d, mask_2d, z_index), where mask_2d can be None.
    """
    image = _squeeze_channel(image)
    mask = _squeeze_channel(mask) if mask is not None else None

    if image.ndim != 3:
        return image, mask, 0  # Return a default z_index for 2D images

    z_index = image.shape[0] // 2
    if mask is not None and mask.ndim == 3 and mask.shape[0] == image.shape[0]:
        # Find slice with the most signal in the mask
        z_sums = np.sum(mask, axis=(1, 2))
        if np.any(z_sums > 0):
            z_index = int(np.argmax(z_sums))

    image_2d = image[z_index]
    mask_2d = mask[z_index] if mask is not None else None

    return image_2d, mask_2d, z_index


def _plot_sample(axes, r: int, c_start: int, data: dict, cols_per_sample: int):
    """Plot a single sample (image, mask, pred) onto the given axes."""
    # Plot Image
    ax_img = axes[r, c_start]
    ax_img.imshow(data["image"], cmap="gray")
    ax_img.set_title(f"img[{data['idx']}]")
    ax_img.axis("off")

    # Plot Mask (if available)
    if data.get("mask") is not None and cols_per_sample > 1:
        ax_mask = axes[r, c_start + 1]
        ax_mask.imshow(data["mask"], cmap="gray", vmin=0, vmax=1)
        ax_mask.set_title(f"msk[{data['idx']}]")
        ax_mask.axis("off")

    # Plot Prediction (if available)
    if data.get("pred") is not None and cols_per_sample > 2:
        # The prediction is now pre-sliced to 2D before being passed in.
        ax_pred = axes[r, c_start + 2]
        ax_pred.imshow(data["pred"], cmap="gray", vmin=0, vmax=1)
        ax_pred.set_title(f"pred[{data['idx']}]")
        ax_pred.axis("off")


def visualize_dataset(
    dataset,
    predictions: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None,
    rows: int = 5,
    samples_per_row: int = 4,
    title: Optional[str] = None,
) -> None:
    """
    Saves or displays a grid of samples from a dataset.

    Args:
        dataset: The dataset to visualize.
        predictions: Optional list of prediction masks to display alongside images and GT masks.
        save_path: Directory to save the output PNG. If None, displays the plot.
        rows: Number of rows in the grid.
        samples_per_row: Number of samples per row.
        title: Figure title and base for the output filename.
    """
    import matplotlib.pyplot as plt
    import os

    if not dataset:
        return

    # Determine grid layout
    _, first_mask, first_pred = _get_sample_data(dataset, 0, predictions)
    has_mask = first_mask is not None
    has_preds = predictions is not None and len(predictions) > 0
    cols_per_sample = 3 if has_mask and has_preds else 2 if has_mask else 1
    cols = cols_per_sample * samples_per_row

    # Select interesting samples to display
    total_to_show = rows * samples_per_row
    selected_samples = []
    for i in range(len(dataset)):
        if len(selected_samples) >= total_to_show:
            break

        image, mask, pred = _get_sample_data(dataset, i, predictions)

        # For datasets with masks, only show samples with signal
        if has_mask and (mask is None or not np.any(mask > 0)):
            continue

        image_2d, mask_2d, z_index = _select_best_z_slice(image, mask)

        sample_data = {"idx": i, "image": image_2d}
        if mask_2d is not None:
            sample_data["mask"] = (mask_2d > 0.5).astype(np.uint8)
        
        if pred is not None:
            # Slice the prediction here using the z_index from the image/mask
            pred_np = _squeeze_channel(pred)
            pred_2d = pred_np[z_index] if pred_np.ndim == 3 else pred_np
            sample_data["pred"] = (pred_2d > 0.5).astype(np.uint8)

        selected_samples.append(sample_data)

    if not selected_samples:
        print("No suitable samples found for visualization.")
        return

    # Create and fill the plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.atleast_2d(axes) # Ensure axes is always a 2D array

    for i, sample_data in enumerate(selected_samples):
        row_idx = i // samples_per_row
        col_start = (i % samples_per_row) * cols_per_sample
        if row_idx >= rows:
            break
        _plot_sample(axes, row_idx, col_start, sample_data, cols_per_sample)

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # Save or show the figure
    if save_path:
        filename = "dataset_preview.png"
        if title:
            safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
            filename = f"{safe_title}.png"
        plt.savefig(os.path.join(save_path, filename))
    else:
        plt.show()

    plt.close(fig)
