import torch.nn.functional as F

#
# Segmentation-only losses (binary or multilabel). All functions share:
# - from_logits: if True, apply sigmoid for overlap metrics and use BCE-with-logits.
# - reduction: one of {"mean", "sum", "none"}.
# - smooth: small constant for numerical stability.
# Shapes: pred, target must match and be (N, ...) with >= 2 dims.
#

def _prepare_inputs(pred, target, *, from_logits: bool):
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")
    if pred.dim() < 2:
        raise ValueError("segmentation losses expect tensors with at least 2 dims: (N, ...)" )

    # Cast to float for computations
    pred = pred.float()
    target = target.float()

    if from_logits:
        pred = pred.sigmoid()

    # Reduce across all dims except batch for global overlap per sample
    reduce_dims = tuple(range(1, pred.dim()))
    return pred.contiguous(), target.contiguous(), reduce_dims


def _reduce(loss, reduction: str):
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Invalid reduction: {reduction}")


def dice_score(
    pred,
    target,
    smooth: float = 1e-6,
    *,
    from_logits: bool = False,
    reduction: str = "mean",
):
    """
    Soft Dice score for segmentation (binary/multilabel).
    - Computes per-sample global Dice (reduces all dims except batch) then applies reduction.
    - If from_logits=True, applies sigmoid to pred.
    """
    pred, target, dims = _prepare_inputs(pred, target, from_logits=from_logits)
    intersection = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return _reduce(dice, reduction)


def dice_loss(
    pred,
    target,
    smooth: float = 1e-6,
    *,
    from_logits: bool = False,
    reduction: str = "mean",
):
    """
    Dice loss for segmentation (binary/multilabel) = 1 - dice_score.
    Uses the same normalization and reduction as dice_score.
    """
    score = dice_score(pred, target, smooth=smooth, from_logits=from_logits, reduction="none")
    loss = 1.0 - score
    return _reduce(loss, reduction)


def dice_bce_loss(
    pred,
    target,
    smooth: float = 1e-6,
    *,
    from_logits: bool = False,
    dice_weight: float = 0.7,
    bce_weight: float = 0.3,
    reduction: str = "mean",
):
    """
    Weighted combination of Dice loss and BCE for segmentation.
    - If from_logits=True: uses BCE-with-logits and sigmoid for Dice.
    """
    # BCE first (on logits or probs depending on flag)
    if from_logits:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
    else:
        bce = F.binary_cross_entropy(pred, target, reduction=reduction)

    dl = dice_loss(pred, target, smooth=smooth, from_logits=from_logits, reduction=reduction)
    return dice_weight * dl + bce_weight * bce


def tversky_score(
    pred,
    target,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
    from_logits: bool = False,
    reduction: str = "mean",
):
    """
    Tversky index (score) for segmentation (binary/multilabel).
    alpha penalizes false positives; beta penalizes false negatives.
    alpha=beta=0.5 approximates Dice.
    """
    pred, target, dims = _prepare_inputs(pred, target, from_logits=from_logits)
    tp = (pred * target).sum(dims)
    fp = (pred * (1.0 - target)).sum(dims)
    fn = ((1.0 - pred) * target).sum(dims)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return _reduce(tversky, reduction)


def tversky_loss(
    pred,
    target,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
    from_logits: bool = False,
    reduction: str = "mean",
):
    """
    Tversky loss = 1 - tversky_score.
    """
    score = tversky_score(
        pred,
        target,
        alpha=alpha,
        beta=beta,
        smooth=smooth,
        from_logits=from_logits,
        reduction="none",
    )
    loss = 1.0 - score
    return _reduce(loss, reduction)


def tversky_bce_loss(
    pred,
    target,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
    tversky_weight: float = 0.7,
    bce_weight: float = 0.3,
    from_logits: bool = False,
    reduction: str = "mean",
):
    """
    Weighted combination of Tversky loss and BCE for segmentation.
    - If from_logits=True: uses BCE-with-logits and sigmoid for Tversky.
    """
    if from_logits:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
    else:
        bce = F.binary_cross_entropy(pred, target, reduction=reduction)

    tl = tversky_loss(
        pred,
        target,
        alpha=alpha,
        beta=beta,
        smooth=smooth,
        from_logits=from_logits,
        reduction=reduction,
    )
    return tversky_weight * tl + bce_weight * bce
