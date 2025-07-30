import numpy as np
from skimage.transform import resize

def stitch_image_xy(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1)):
    """
    Reconstructs the full 3D image from patches using weighted averaging.
    
    Args:
        patches (list or np.ndarray): List or array of 3D patches.
        positions (list or np.ndarray): Corresponding (z, y, x) positions.
        original_shape (tuple): Shape of the original image (D, H, W).
        patch_size (tuple): Size of each patch (cd, ch, cw).
        resize_factor (tuple): Resize factor used during patch extraction (default: no resize).
        
    Returns:
        np.ndarray: Reconstructed image of shape original_shape.
    """
    reconstruction = np.zeros(original_shape, dtype=np.float32)
    weight = np.zeros(original_shape, dtype=np.float32)
    pd, ph, pw = patch_size

    for patch, (d, h, w) in zip(patches, positions):
        # Resize patch back to original patch size if it was resized
        if resize_factor != (1, 1, 1):
            patch = resize(
                patch,
                (pd, ph, pw),
                order=1,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.float32)

        # Add patch to reconstruction
        reconstruction[d:d+pd, h:h+ph, w:w+pw] += patch
        weight[d:d+pd, h:h+ph, w:w+pw] += 1

    # Avoid division by zero
    weight[weight == 0] = 1
    reconstruction /= weight

    return reconstruction

def stitch_image_z(reconstruction: np.ndarray, prev_z_slices: np.ndarray, threshold=0.5):
    """
    Blends overlapping Z slices across volumes.
    
    Args:
        reconstruction (np.ndarray): Current 3D patch volume (Z, Y, X).
        prev_z_slices (np.ndarray or None): Last Z-overlap slices from previous volume.
        z_overlay (int): Number of Z slices to blend.
        threshold (float): Threshold for binary mask.

    Returns:
        binary_mask (np.ndarray): Thresholded binary mask of shape (Z, Y, X).
    """
    if prev_z_slices is None:
        return (reconstruction > threshold).astype(np.uint8)
    else:
        if len(prev_z_slices.shape) < 3:
            z_overlay = 1
        else:
            z_overlay = prev_z_slices.shape[0]
        
        if prev_z_slices.shape != reconstruction[:z_overlay].shape:
            raise ValueError(f"Shape mismatch between previous and current Z slices.")
        reconstruction[:z_overlay] = (
            reconstruction[:z_overlay] + prev_z_slices
        ) / 2

    return (reconstruction > threshold).astype(np.uint8)

def stitch_image(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1), prev_z_slices=None, z_overlay=0):
    """
    Stitch patches into a full image with optional Z slice blending.
    
    Args:
        patches (list or np.ndarray): List or array of 3D patches.
        positions (list or np.ndarray): Corresponding (z, y, x) positions.
        original_shape (tuple): Shape of the original image (D, H, W).
        patch_size (tuple): Size of each patch (cd, ch, cw).
        resize_factor (tuple): Resize factor used during patch extraction.
        prev_z_slices (np.ndarray or None): Last Z slices from previous volume.
        z_overlay (int): Number of Z slices to blend.

    Returns:
        np.ndarray: Reconstructed full image.
    """
    reconstruct_xy = stitch_image_xy(patches, positions, original_shape, patch_size, resize_factor)
    reconstruction = stitch_image_z(reconstruct_xy, prev_z_slices) # type: ignore
    
    if z_overlay > 0:
        return reconstruction[:-z_overlay], reconstruction[-z_overlay:]
    else:
        return reconstruction, None