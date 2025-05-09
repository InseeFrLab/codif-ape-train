from src.utils.evaluation import get_confidence_score as confidence_score_func


def get_confidence_score(predictions):
    sorted_confidence, sorted_confidence_indices = predictions.sort(descending=True, dim=-1)

    return confidence_score_func(sorted_confidence)


def find_y_position(sorted_pred_idx, y):
    mask = (sorted_pred_idx == y.unsqueeze(-1)).int().argmax(dim=-1)  # Find position of y
    return mask
