import torch
from tqdm import tqdm
import numpy as np

class Inferencer:
    def __init__(self, model: torch.nn.Module, device=None):
        """
        Args:
            model (torch.nn.Module): The trained model for inference.
            device (torch.device): The device to run inference on (e.g., torch.device("cuda")).
        """
        self.model = model.to(device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def eval(self, inference_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Run inference on the entire dataset using the provided DataLoader.

        Args:
            inference_loader (torch.utils.data.DataLoader): DataLoader with input samples.

        Returns:
            np.ndarray: Model predictions concatenated as a NumPy array.
        """
        outputs = []
        progress_bar = tqdm(inference_loader, desc=f"Inferencing ", leave=False)

        with torch.no_grad():
            for inputs in progress_bar:
                # If dataloader returns (input, target), ignore target
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
                    
                inputs = inputs.to(self.device)
                preds = self.model(inputs)
                preds = preds.squeeze(1).detach().cpu().numpy()

                outputs.append(preds)

        return np.concatenate(outputs, axis=0)  # [N, ...] where N = total samples