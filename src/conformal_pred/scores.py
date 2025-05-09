import inspect

import torch

from .utils import find_y_position


def get_adaptive_score(x, y):
    n_samples = len(y)

    # Get sorted predictions and indices
    sorted_pred, sorted_pred_idx = x.sort(dim=-1, descending=True)
    sorted_pred_cumsum = sorted_pred.cumsum(dim=-1)  # Cumulative sum

    mask = find_y_position(sorted_pred_idx, y)  # ground truth position in the sorted probabilities
    scores = sorted_pred_cumsum[
        torch.arange(n_samples), mask
    ]  # Cumulative sum up of the biggest scores till y

    return scores


def high_proba_score(x, y):
    scores = 1 - x[list(range(len(x))), y]  # 1 - p(y|x)
    return scores


def get_position_score(x, y):
    sorted_pred, sorted_pred_idx = x.sort(dim=1, descending=True)
    mask = find_y_position(sorted_pred_idx, y)  # ground truth position in the sorted probabilities
    return mask


def _validate_score_func(fn):
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if len(params) != 2:
        raise TypeError("score_func must take exactly two arguments: (predictions, labels)")

    # Try a dummy call
    try:
        dummy_x = torch.softmax(torch.randn(10, 5), dim=1)
        dummy_y = torch.randint(0, 5, (10,))
        out = fn(dummy_x, dummy_y)

        if not isinstance(out, torch.Tensor):
            raise TypeError("score_func must return a torch.Tensor.")
        if out.shape != (10,):
            raise ValueError("score_func must return a 1D tensor of shape (n_samples,)")
    except Exception as e:
        raise TypeError(f"score_func failed on dummy input: {e}")
