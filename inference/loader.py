from monai.data.dataset import Dataset

class MicroscopyDataset2D(Dataset):
    """
    Custom PyTorch Dataset for 2D microscopy image segmentation.
    Supports MONAI or custom transforms.
    """
    def __init__(self, patch_dicts, transform=None):
        """
        Args:
            patch_dicts (list of dict): Each dict has keys 'image' and 'mask'.
            transform (callable, optional): Transform to apply on a sample dict.
        """
        self.patch_dicts = patch_dicts
        self.transform = transform
        
    def __len__(self):
        return len(self.patch_dicts)

    def __getitem__(self, idx):
        sample = self.patch_dicts[idx]

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
            
        image = sample["image"]

        # Manually add channel dimension if missing
        if image.ndim == 2:  # shape: [H, W] → [1, H, W]
            image = image.unsqueeze(0)

        return image

class MicroscopyDataset3D(Dataset):
    """
    Custom PyTorch Dataset for 3D microscopy image segmentation.
    Supports MONAI or custom transforms.
    """
    def __init__(self, patch_dicts, transform=None):
        """
        Args:
            patch_dicts (list of dict): Each dict has keys 'image' and 'mask'.
            transform (callable, optional): Transform to apply on a sample dict.
        """
        self.patch_dicts = patch_dicts
        self.transform = transform
        
    def __len__(self):
        return len(self.patch_dicts)

    def __getitem__(self, idx):
        sample = self.patch_dicts[idx]

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
            
        image = sample["image"]

        # Manually add channel dimension if missing
        if image.ndim == 3:  # shape: [D, H, W] → [1, D, H, W]
            image = image.unsqueeze(0)

        return image