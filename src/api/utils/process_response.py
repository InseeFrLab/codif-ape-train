from fastapi import HTTPException

from ..models import Prediction, PredictionResponse


def process_response(
    predictions: tuple,
    liasse_nb: int,
    nb_echos_max: int,
    prob_min: float,
    libs: dict,
) -> PredictionResponse:
    """
    Process model
    predictions into a structured response.
    """
    labels, probs = predictions
    pred_labels = labels[liasse_nb]
    pred_probs = probs[liasse_nb]

    valid_preds = []
    mask = pred_probs >= prob_min
    valid_predicted_class = pred_labels[mask]
    valid_predicted_confidence = pred_probs[mask]
    valid_preds.append(tuple(zip(valid_predicted_class, valid_predicted_confidence)))

    if not valid_preds:
        raise HTTPException(
            status_code=400,
            detail="No prediction exceeds the minimum probability threshold.",
        )

    response_data = {
        str(i + 1): Prediction(
            code=label,
            probabilite=float(prob),
            libelle=libs[label],
        )
        for i, (label, prob) in enumerate(valid_preds[0])
    }

    confidence_score = pred_probs[0] - pred_probs[1]
    response_data["IC"] = confidence_score

    return PredictionResponse(response_data)
