"""Real-time estimates: what a user could have known at each month end.

The published series is re-estimated on the full sample every month, so its
history is revised continually. Here the model is estimated on an expanding
window ending at each month t, factors included, and only the end-of-window
decomposition is kept. The collection of those end points is the real-time
term premium; the difference from the full-sample estimate is its revision.
Each month's fitted model is also kept, because the policy-day analysis prices
daily yields with the model a user had at the end of the previous month.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from acm_term_premium.config import ResearchConfig
from acm_term_premium.models.acm import AcmModel
from acm_term_premium.models.configured import fit_model


@dataclass(frozen=True)
class RealtimeOutput:
    term_premium: pd.DataFrame  # rows: month end t, columns: maturity (months), value known at t
    risk_neutral: pd.DataFrame
    fitted: pd.DataFrame
    unconstrained_term_premium: pd.DataFrame  # same, with no bound on the VAR's roots
    diagnostics: pd.DataFrame  # spectral radius of the OLS VAR and whether the bound bound
    models: dict[pd.Timestamp, AcmModel]


def run(
    config: ResearchConfig,
    curve: pd.DataFrame,
    months: list[int],
    progress: Callable[[str], None] | None = None,
) -> RealtimeOutput:
    m = config.model
    out: dict[str, dict[pd.Timestamp, np.ndarray]] = {
        "term_premium": {},
        "risk_neutral": {},
        "fitted": {},
        "unconstrained_term_premium": {},
    }
    diagnostics: dict[pd.Timestamp, tuple[float, bool]] = {}
    models: dict[pd.Timestamp, AcmModel] = {}
    for end in range(config.realtime.min_months, curve.shape[0] + 1):
        window = curve.iloc[:end]
        model = fit_model(window, m)
        last = window.iloc[[-1]]
        decomposition = model.decompose(last, months)
        date = pd.Timestamp(window.index[-1])
        for key, frame in decomposition.items():
            out[key][date] = frame.to_numpy()[0]
        if model.constrained:
            free = fit_model(window, m, bounded=False)
            raw = free.decompose(last, months)["term_premium"].to_numpy()[0]
        else:
            raw = decomposition["term_premium"].to_numpy()[0]
        out["unconstrained_term_premium"][date] = raw
        diagnostics[date] = (model.spectral_radius, model.constrained)
        models[date] = model
        if progress is not None and end % 120 == 0:
            progress(f"real-time estimates through {date.date()}")
    frames = {
        key: pd.DataFrame.from_dict(values, orient="index", columns=months)
        for key, values in out.items()
    }
    diag = pd.DataFrame.from_dict(
        diagnostics, orient="index", columns=["spectral_radius", "constrained"]
    )
    return RealtimeOutput(
        frames["term_premium"],
        frames["risk_neutral"],
        frames["fitted"],
        frames["unconstrained_term_premium"],
        diag,
        models,
    )
