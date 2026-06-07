"""Annual medical cost predictor and what-if simulator.

Public surface:

* `PredictionFeatures`, `CostPrediction`: pydantic schemas for the model.
* `CostPredictor`: quantile-regression ensemble (10th, 50th, 90th percentile).
* `sweep`: vary one feature across values and compare predictions.
* `apply_plan_to_annual_spend`: bridge predicted charges into the
  benefits engine to estimate annual out-of-pocket.
"""

from gauge.predictor.annual_cost import (
    AnnualPlanShare,
    apply_plan_to_annual_spend,
)
from gauge.predictor.dataset import generate_synthetic_dataset, load_dataset
from gauge.predictor.model import CostPrediction, CostPredictor
from gauge.predictor.schemas import PredictionFeatures
from gauge.predictor.whatif import WhatIfPoint, WhatIfResponse, sweep

__all__ = [
    "AnnualPlanShare",
    "CostPrediction",
    "CostPredictor",
    "PredictionFeatures",
    "WhatIfPoint",
    "WhatIfResponse",
    "apply_plan_to_annual_spend",
    "generate_synthetic_dataset",
    "load_dataset",
    "sweep",
]
