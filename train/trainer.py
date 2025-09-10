import torch
import torch.optim as optim
import os
import logging
from tqdm import tqdm

from train.loss import dice_bce_loss, dice_loss, tversky_bce_loss, tversky_loss

# Initialize logging
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, device=None, lr=0.001, model_name="best_model", save_path="./"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)

        # Store history of metrics
        self.metrics_history = {
            "loss": {
                "train": [],
                "val": [],
            },
        }

        self.best_val_loss = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for images, masks in progress_bar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = dice_bce_loss(outputs, masks)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(self.train_loader)
        self.metrics_history["loss"]["train"].append(avg_loss)
        return avg_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)

        with torch.no_grad():
            for images, masks in progress_bar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                outputs = self.model(images)
                loss = dice_bce_loss(outputs, masks)
                val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = val_loss / len(self.val_loader)
        self.metrics_history["loss"]["val"].append(avg_loss)
        return avg_loss

    def train(self, epochs=30):
        for epoch in range(epochs):
            print("\n")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f}")

            val_loss = self.validate_epoch(epoch)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
                model_path = os.path.join(self.save_path, f"{self.model_name}.pth")
                torch.save(self.model, model_path)
                logger.info("✅ Model Saved!")

            # Save figure after each epoch with error handling
            try:
                self.save_figure(metric_name="loss", epoch=epoch + 1)
            except Exception as e:
                logger.warning("Could not save figure at epoch %d: %s", epoch + 1, str(e))
            
    def save_figure(self, metric_name="loss", epoch=None):
        """Save the training and validation curves for a given metric.

        If `epoch` is provided, also save a uniquely named snapshot for that epoch
        to avoid issues when a viewer has the last image open on Windows.
        """
        # Use a headless backend to avoid Tkinter threading errors on Windows
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        if metric_name not in self.metrics_history:
            logger.warning("Metric '%s' not found in metrics_history. Available keys: %s",
                        metric_name, list(self.metrics_history.keys()))
            return

        metric_data = self.metrics_history[metric_name]
        if not metric_data["train"] or not metric_data["val"]:
            logger.warning("No data available for metric '%s'. Skipping plot.", metric_name)
            return

        plt.figure(figsize=(8, 6))
        plt.plot(metric_data["train"], label=f"Train {metric_name.capitalize()}")
        plt.plot(metric_data["val"], label=f"Validation {metric_name.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.title(f"Training and Validation {metric_name.capitalize()}")
        plt.legend()
        plt.grid(True)

        base_name = f"{self.model_name}-{metric_name}_curve"
        latest_path = os.path.join(self.save_path, f"{base_name}.png")

        # Always try to update the latest figure; if it's open, fall back silently
        try:
            plt.savefig(latest_path)
        except Exception as e:
            logger.warning(
                "Could not update latest figure '%s': %s. Will try epoch-specific name.",
                latest_path, str(e)
            )
            
        plt.close()
