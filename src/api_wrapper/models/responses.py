from typing import Dict, Union

from pydantic import BaseModel, RootModel


class Prediction(BaseModel):
    code: str
    probabilite: float
    libelle: str


class PredictionResponse(BaseModel):
    predictions: List[Prediction]
    IC: float
    MLversion: Optional[str] = None
    pass
