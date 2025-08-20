from typing import Dict, Union

from pydantic import BaseModel, RootModel


class Prediction(BaseModel):
    code: str
    probabilite: float
    libelle: str


class PredictionResponse(RootModel[Dict[str, Union[Prediction, float, str]]]):
    pass
