from typing import Any, Dict, Mapping

from pydantic import BaseModel, RootModel, model_validator


class Prediction(BaseModel):
    code: str
    probabilite: float
    libelle: str


class PredictionResponse(RootModel[Dict[str, Any]]):
    """
    Compact RootModel that accepts the flat response shape:
      { "1": {...}, "2": {...}, ..., "IC": float, "MLversion": str }

    - IC is required and coerced to float
    - MLversion is optional and coerced to str if present
    - numeric keys are validated as Prediction
    - any unexpected keys cause validation error
    """
    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, Mapping):
            raise TypeError("PredictionResponse: expected a dict/mapping")

        # IC (required)
        try:
            ic = float(data["IC"])
        except KeyError:
            raise ValueError("PredictionResponse: missing required key 'IC'")
        except Exception as e:
            raise ValueError(f"PredictionResponse: 'IC' not convertible to float: {e}") from e

        # optional MLversion
        ml = data.get("MLversion")
        mlversion = None if ml is None else str(ml)

        # collect numeric keys and detect unexpected keys
        preds: Dict[str, Any] = {}
        allowed = {"IC", "MLversion"}
        for k, v in data.items():
            if isinstance(k, str) and k.isdigit():
                preds[k] = Prediction.model_validate(v).model_dump()
                allowed.add(k)

        extra = set(data.keys()) - allowed
        if extra:
            raise ValueError(f"PredictionResponse: unexpected keys: {sorted(extra)}")

        # return flat dict matching the historic JSON shape
        return {**preds, "IC": ic, **({"MLversion": mlversion} if "MLversion" in data else {})}
