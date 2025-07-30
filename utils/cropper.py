import numpy as np
from skimage.transform import resize
    
def extract_patches(
    array: np.ndarray,
    patch_size=(16, 64, 64),
    overlay=(2, 4, 4),
    resize_factor=(1, 1, 1),
    return_positions=False
):
    """
    Extracts overlapping 3D patches from an input volume with optional resizing.

    The function slides a window of size `patch_size` over the input `array`, stepping by 
    `patch_size - overlay` along each dimension. Each patch can optionally be resized 
    according to `resize_factor`.

    Args:
        array (np.ndarray): 3D input volume of shape (Z, Y, X).
        patch_size (tuple): Size of each patch (depth, height, width).
        overlay (tuple): Number of overlapping voxels between adjacent patches (z, y, x).
        resize_factor (tuple): Resize factor (z, y, x) for each extracted patch. Default is (1, 1, 1) = no resize.
        return_positions (bool): If True, also returns the (z, y, x) starting position of each patch.

    Returns:
        np.ndarray: Array of extracted (and optionally resized) 3D patches.
        np.ndarray (optional): Array of (z, y, x) positions for each patch if `return_positions=True`.
    """
    D, H, W = array.shape
    patches, positions = [], []

    for z in range(0, D, patch_size[0] - overlay[0]):
        start_z = min(z, D - patch_size[0])
        for y in range(0, H, patch_size[1] -overlay[1]):
            start_y = min(y, H - patch_size[1])
            for x in range(0, W, patch_size[2] - overlay[2]):
                start_x = min(x, W - patch_size[2])
                
                patch = array[
                    start_z:start_z + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2]
                ]
                
                # Resize if needed
                if resize_factor != (1, 1, 1):
                    new_shape = (
                        int(patch.shape[0] * resize_factor[0]),
                        int(patch.shape[1] * resize_factor[1]),
                        int(patch.shape[2] * resize_factor[2])
                    )
                    patch = resize(
                        patch, new_shape,
                        order=1,  # linear interpolation
                        mode='reflect',
                        anti_aliasing=True,
                        preserve_range=True
                    ).astype(patch.dtype)
                
                patches.append(patch)
                positions.append((start_z, start_y, start_x))

    if return_positions:
        return np.array(patches), np.array(positions)
    else:
        return np.array(patches)

def balance_patches(image_patches, mask_patches):
    """
    Balances a dataset of image and mask patches by ensuring equal numbers of
    positive (signal-containing) and negative (background-only) samples.

    A patch is considered "positive" if the corresponding mask contains any non-zero values.

    Args:
        image_patches (list): List of image patch arrays.
        mask_patches (list): List of corresponding binary mask patch arrays.

    Returns:
        tuple:
            np.ndarray: Balanced list of image patches.
            np.ndarray: Balanced list of corresponding mask patches.

    Note:
        - The output will have `2 * min(n_positive, n_negative)` total patches.
        - The result is shuffled randomly.
    """
    positive_patches = []
    negative_patches = []
    
    for img_patch, mask_patch in zip(image_patches, mask_patches):
        if np.sum(mask_patch) > 0:  # If signal is present
            positive_patches.append((img_patch, mask_patch))
        else:
            negative_patches.append((img_patch, mask_patch))

    # Balance dataset
    min_size = min(len(positive_patches), len(negative_patches))
    
    positive_patches = positive_patches[:min_size]
    negative_patches = negative_patches[:min_size]

    balanced_patches = positive_patches + negative_patches
    np.random.shuffle(balanced_patches)  # Shuffle dataset

    image_patches, mask_patches = zip(*balanced_patches)
    return np.array(image_patches), np.array(mask_patches)

def extract_training_batches(
    image:np.ndarray,
    mask: np.ndarray,
    patch_size=(16, 64, 64),
    overlay=(2, 4, 4),
    resize_factor=(1, 1, 1),
    balance=True
):
    """
    Extracts training patches from a 3D image and corresponding mask, with optional balancing
    of positive and negative samples.

    This function extracts patches from both the image and mask using the same settings, then
    optionally balances them based on signal presence in the mask.

    Args:
        image (np.ndarray): 3D image array (Z, Y, X).
        mask (np.ndarray): 3D mask array (same shape as image).
        patch_size (tuple): Size of each patch (depth, height, width).
        overlay (tuple): Number of overlapping voxels between patches (z, y, x).
        resize_factor (tuple): Factor to resize each patch (z, y, x).
        balance (bool): Whether to balance the number of signal and background patches.

    Returns:
        tuple:
            np.ndarray: Image patches.
            np.ndarray: Corresponding mask patches.

    Raises:
        ValueError: If `image` and `mask` shapes do not match.
    """
    
    if (image.shape != mask.shape):
        raise ValueError("Image and mask must have the same shape.")
    
    image_patches = extract_patches(array=image, patch_size=patch_size, overlay=overlay, resize_factor=resize_factor)
    mask_patches = extract_patches(array=mask, patch_size=patch_size, overlay=overlay, resize_factor=resize_factor)
    
    if balance:
        return balance_patches(image_patches, mask_patches)
    else:
        return image_patches, mask_patches