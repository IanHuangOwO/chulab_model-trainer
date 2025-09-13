import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union
from tqdm import tqdm

from train.loss import (
    dice_score, dice_loss, dice_bce_loss, 
    tversky_score, tversky_loss, tversky_bce_loss
)

# Initialize logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Metrics configuration (outside the class as requested)
# Define metrics as a mapping from name -> function(outputs, targets) -> value.
# The value can be a torch.Tensor (for losses to backprop) or a float for
# logging-only metrics. The trainer will pick PRIMARY_LOSS_KEY for backward().
# -----------------------------------------------------------------------------
MetricFn = Callable[[torch.Tensor, torch.Tensor], Union[torch.Tensor, float]]

PRIMARY_LOSS_KEY: str = "dice_bce_loss"
METRICS_TO_COMPUTE: Dict[str, MetricFn] = {
    # Trainable loss used for backward
    "dice_bce_loss": lambda outputs, targets: dice_bce_loss(outputs, targets),
    "dice_score": lambda outputs, targets: dice_score(outputs, targets),
    "bce_score": lambda outputs, targets: F.binary_cross_entropy(outputs.float(), targets.float()),
}


# (helper removed; metrics are computed inline using PRIMARY_LOSS_KEY)

class Trainer:
    """Simple, readable training loop with pluggable metrics.

    - Metrics are defined by the module-level METRICS_TO_COMPUTE dict.
    - The metric keyed by PRIMARY_LOSS_KEY is used for backprop.
    - History is recorded per metric for train/val and used for plotting.
    """

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

        # Store history of metrics (auto-create for all registered metrics)
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {
            name: {"train": [], "val": []} for name in METRICS_TO_COMPUTE
        }

        self.best_val_loss = float("inf")

    def _run_epoch(self, epoch: int, *, train: bool) -> float:
        """Run a single epoch and return the average primary loss.

        When train=True, performs backprop; otherwise uses no_grad().
        Also updates self.metrics_history for the corresponding split.
        """
        self.model.train(mode=train)
        loader = self.train_loader if train else self.val_loader
        desc = f'{"Training" if train else "Validating"} Epoch {epoch+1}'
        sums: Dict[str, float] = {m: 0.0 for m in METRICS_TO_COMPUTE.keys()}
        n_batches = max(1, len(loader))

        context = torch.enable_grad if train else torch.no_grad
        with context():
            progress = tqdm(loader, desc=desc, leave=False)
            for images, masks in progress:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(images)

                # Compute primary (trainable) loss directly from registry
                loss_val = METRICS_TO_COMPUTE[PRIMARY_LOSS_KEY](outputs, masks)
                if not isinstance(loss_val, torch.Tensor):
                    raise TypeError(
                        f"Metric '{PRIMARY_LOSS_KEY}' must return a torch.Tensor (got {type(loss_val)!r})."
                    )

                if train:
                    loss_val.backward()
                    self.optimizer.step()

                # Collect batch metrics (avoid recomputing primary)
                batch_metrics: Dict[str, float] = {PRIMARY_LOSS_KEY: float(loss_val.item())}
                for name, fn in METRICS_TO_COMPUTE.items():
                    if name == PRIMARY_LOSS_KEY:
                        continue
                    try:
                        v = fn(outputs, masks)
                        batch_metrics[name] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                    except Exception as e:
                        logger.warning("Metric '%s' computation failed: %s", name, str(e))

                for k, v in batch_metrics.items():
                    sums[k] = sums.get(k, 0.0) + v

                progress.set_postfix({k: f"{batch_metrics[k]:.4f}" for k in batch_metrics})

        split = "train" if train else "val"
        for m in METRICS_TO_COMPUTE.keys():
            avg = sums.get(m, 0.0) / n_batches
            if m in self.metrics_history:
                self.metrics_history[m][split].append(avg)

        return sums.get(PRIMARY_LOSS_KEY, 0.0) / n_batches

    def train_epoch(self, epoch: int) -> float:
        return self._run_epoch(epoch, train=True)

    def validate_epoch(self, epoch):
        return self._run_epoch(epoch, train=False)

    def train(self, epochs=30):
        for epoch in range(epochs):
            print("\n")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f}")
            # Log all train metrics collected this epoch
            try:
                train_metrics = {
                    name: self.metrics_history[name]["train"][-1]
                    for name in METRICS_TO_COMPUTE
                    if self.metrics_history.get(name, {}).get("train")
                }
                logger.info(
                    "Train metrics: %s",
                    ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()),
                )
            except Exception:
                pass

            val_loss = self.validate_epoch(epoch)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            # Log all val metrics collected this epoch
            try:
                val_metrics = {
                    name: self.metrics_history[name]["val"][-1]
                    for name in METRICS_TO_COMPUTE
                    if self.metrics_history.get(name, {}).get("val")
                }
                logger.info(
                    "Val metrics: %s",
                    ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                )
            except Exception:
                pass
            
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
                model_path = os.path.join(self.save_path, f"{self.model_name}.pth")
                torch.save(self.model, model_path)
                logger.info("✅ Model Saved!")

            # Save figure after each epoch with error handling
            try:
                self.save_figure(metric_name="all", epoch=epoch + 1)
            except Exception as e:
                logger.warning("Could not save figure at epoch %d: %s", epoch + 1, str(e))
            
    def save_figure(self, metric_name="all", epoch=None):
        """Save training/validation curves.

        - If metric_name == "all": save a single figure with subplots for all metrics in metrics_history.
        - Else: save a single-plot figure for the specified metric.
        If `epoch` is provided, also save a uniquely named snapshot for that epoch
        to avoid issues when a viewer has the last image open on Windows.
        """
        # Use a headless backend to avoid Tkinter threading errors on Windows
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        metrics_to_plot = []
        if metric_name == "all":
            metrics_to_plot = [k for k, v in self.metrics_history.items() if v["train"] and v["val"]]
            if not metrics_to_plot:
                logger.warning("No metrics with data to plot.")
                return
        else:
            if metric_name not in self.metrics_history:
                logger.warning(
                    "Metric '%s' not found in metrics_history. Available keys: %s",
                    metric_name, list(self.metrics_history.keys())
                )
                return
            if not self.metrics_history[metric_name]["train"] or not self.metrics_history[metric_name]["val"]:
                logger.warning("No data available for metric '%s'. Skipping plot.", metric_name)
                return
            metrics_to_plot = [metric_name]

        # Setup figure with max 3 rows; fill column-first
        import math
        n = len(metrics_to_plot)
        rows = min(3, n)
        cols = math.ceil(n / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4.5 * rows))
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]  # type: ignore
        axes = np.array(axes).reshape(rows, cols)

        # Plot each metric (column-first ordering)
        for idx, m in enumerate(metrics_to_plot):
            r = idx % rows
            c = idx // rows
            ax = axes[r, c]
            data = self.metrics_history[m]
            ax.plot(data["train"], label=f"Train {m}")
            ax.plot(data["val"], label=f"Val {m}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(m)
            ax.set_title(f"{m.capitalize()} over Epochs")
            ax.grid(True)
            ax.legend()

        # Hide any unused subplots (column-first ordering)
        for j in range(n, rows * cols):
            r = j % rows
            c = j // rows
            fig.delaxes(axes[r, c])

        base_name = f"{self.model_name}-metrics_curve" if metric_name == "all" else f"{self.model_name}-{metric_name}_curve"
        latest_path = os.path.join(self.save_path, f"{base_name}.png")

        # Always try to update the latest figure; if it's open, fall back silently
        try:
            fig.tight_layout()
            plt.savefig(latest_path)
        except Exception as e:
            logger.warning(
                "Could not update latest figure '%s': %s. Will try epoch-specific name.",
                latest_path, str(e)
            )
        finally:
            plt.close(fig)
