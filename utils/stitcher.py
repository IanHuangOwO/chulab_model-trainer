import numpy as np
from skimage.transform import resize

def stitch_image_xy(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1)):
    """
    Reconstructs a full 3D volume from overlapping patches using weighted averaging.

    Args:
        patches (list or np.ndarray): List or array of 3D patches (Z, Y, X).
        positions (list or np.ndarray): Corresponding positions (z, y, x) for placing each patch.
        original_shape (tuple): Shape of the full output volume (depth, height, width).
        patch_size (tuple): Size of each patch (depth, height, width).
        resize_factor (tuple): Factor used to resize patches during extraction. If not (1, 1, 1), patches will be resized back.

    Returns:
        np.ndarray: Full reconstructed volume of shape `original_shape`, merged using averaging in overlapping regions.
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
    Performs blending across overlapping Z slices between consecutive volume chunks and returns a binary mask.

    Args:
        reconstruction (np.ndarray): Reconstructed 3D volume (Z, Y, X) for the current chunk.
        prev_z_slices (np.ndarray or None): Overlapping Z slices from the previous chunk. If None, no blending is performed.
        threshold (float): Threshold for binarizing the final output mask.

    Returns:
        np.ndarray: Binary mask (uint8) after Z-slice blending and thresholding, shape (Z, Y, X).
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
    Reconstructs the full 3D volume from patches and blends overlapping Z slices across chunks.

    This function combines XY-plane patch stitching and optional Z-slice blending between chunks.
    It also returns the last few Z slices (if `z_overlay > 0`) to be used for blending with the next chunk.

    Args:
        patches (list or np.ndarray): List or array of 3D patches (Z, Y, X).
        positions (list or np.ndarray): Corresponding (z, y, x) positions for each patch.
        original_shape (tuple): Shape of the full volume (depth, height, width).
        patch_size (tuple): Size of each patch (depth, height, width).
        resize_factor (tuple): Resize factor used during patch extraction.
        prev_z_slices (np.ndarray or None): Overlapping Z slices from the previous volume chunk.
        z_overlay (int): Number of Z slices at the end of the current chunk to save for the next blend.

    Returns:
        tuple:
            - np.ndarray: Binary mask of the reconstructed volume (Z, Y, X), excluding the last `z_overlay` slices.
            - np.ndarray or None: The last `z_overlay` slices of the reconstructed volume for use in the next call,
              or None if `z_overlay == 0`.
    """
    reconstruct_xy = stitch_image_xy(patches, positions, original_shape, patch_size, resize_factor)
    reconstruction = stitch_image_z(reconstruct_xy, prev_z_slices) # type: ignore
    
    if z_overlay > 0:
        return reconstruction[:-z_overlay], reconstruction[-z_overlay:]
    else:
        return reconstruction, None