"""Pydantic schemas shared across the predictor stack.

Field names and value spaces mirror the well-known Kaggle insurance.csv
schema so a real dataset can be dropped in without translation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Sex = Literal["male", "female"]
SmokerStatus = Literal["yes", "no"]
# Census regions match what MEPS uses. The Kaggle CSV's cardinal regions
# (northwest, southwest) are mapped onto these at load time.
Region = Literal["northeast", "midwest", "south", "west"]

# Features the model accepts. Kept tight: explicit numeric ranges so bad
# inputs surface as 422s at the API layer instead of garbage predictions.


class PredictionFeatures(BaseModel):
    """Inputs to the annual cost predictor."""

    model_config = ConfigDict(frozen=True)

    age: int = Field(ge=0, le=120)
    sex: Sex
    bmi: float = Field(gt=5.0, lt=80.0)
    children: int = Field(ge=0, le=20)
    smoker: SmokerStatus
    region: Region
