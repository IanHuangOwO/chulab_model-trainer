from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union
from monai.data.dataset import Dataset

class MicroscopyDataset(Dataset):
    """
    Unified dataset for 2D/3D microscopy segmentation for both training and inference.

    - Uses dict-based samples compatible with MONAI transforms: {"image", ["mask"]}.
    - Adds a channel dimension if missing (C=1) based on `spatial_dims`.
    - Returns (image, mask) when `with_mask=True`, else returns image only.

    Args:
        patch_dicts: List of sample dicts containing at least key "image" and
            optionally key "mask" when `with_mask=True`.
        transform: Optional transform (e.g., MONAI Compose) applied to the sample dict.
        spatial_dims: 2 or 3 indicating 2D or 3D images.
        with_mask: Whether to return the mask alongside the image.
    """

    def __init__(
        self,
        patch_dicts: List[Dict[str, Union["object"]]],
        transform: Optional[Callable] = None,
        spatial_dims: int = 3,
        with_mask: bool = True,
    ) -> None:
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

        self.patch_dicts = patch_dicts
        self.transform = transform
        self.spatial_dims = spatial_dims
        self.with_mask = with_mask

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.patch_dicts)

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.patch_dicts[idx]

        # Apply transform first to keep behavior consistent with existing code
        if self.transform is not None:
            sample = self.transform(sample)

        image = sample["image"]

        # Add channel dimension if missing: [H,W] -> [1,H,W] for 2D, [D,H,W] -> [1,D,H,W] for 3D
        if hasattr(image, "ndim"):
            if self.spatial_dims == 2 and image.ndim == 2:
                image = image.unsqueeze(0)
            elif self.spatial_dims == 3 and image.ndim == 3:
                image = image.unsqueeze(0)

        if not self.with_mask:
            return image

        mask = sample.get("mask", None)
        if mask is None:
            raise KeyError("Sample is missing required key 'mask' but with_mask=True")

        if hasattr(mask, "ndim"):
            if self.spatial_dims == 2 and mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif self.spatial_dims == 3 and mask.ndim == 3:
                mask = mask.unsqueeze(0)

        return image, mask

