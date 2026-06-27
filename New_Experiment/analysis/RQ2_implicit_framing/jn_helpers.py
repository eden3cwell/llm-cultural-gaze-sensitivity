"""
jn_helpers.py — polynomial Johnson-Neyman (floodlight) shared across RQ2/RQ3.

Generalises JN to a polynomial moderator: the simple effect of `condition`
on the DV is evaluated across the fixation grid, with its standard error from
the model covariance matrix, and the region of significance is found
numerically (where |t(x)| >= t_crit) rather than via closed-form roots.

The interaction is fitted at the RQ1 Ferguson-retained polynomial degree for
the given (model, DV), so JN localisation corresponds to the reported curves.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import FIX_RANGE, ALPHA


def _poly_terms(x, k):
    """Return [x, x^2, ..., x^k] stacked columns for scalar or array x.
    x is expected scaled to [0,1]."""
    x = np.asarray(x, float)
    return np.column_stack([x ** j for j in range(1, k + 1)])


def _region_label(grid, sig_mask):
    """Human-readable region description from a boolean significance mask."""
    if sig_mask.all():
        return "Significant across entire range [0, 100]"
    if not sig_mask.any():
        return "Not significant anywhere in [0, 100]"
    # find contiguous significant segments
    idx = np.where(sig_mask)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, breaks + 1)
    spans = [f"{grid[s[0]]:.1f}-{grid[s[-1]]:.1f}%" for s in segments]
    return "Significant for fixation " + ", ".join(spans)


def get_effect_curve(
    df: pd.DataFrame,
    treatment_cond: str,
    reference_cond: str,
    dv: str = "engagement_score",
    degree: int = 1,
    alpha: float = ALPHA,
) -> tuple | None:
    """
    Fit  DV ~ cond_bin * poly(fixation, degree)  and return
    (effect_curve, se_curve, jn_result) over FIX_RANGE.

    effect_curve[i] is the simple effect (treatment - reference) at fixation
    grid point i; jn_result carries the significance mask, region label, and
    the fixation boundaries where significance changes.
    """
    subset = df[df["condition"].isin([reference_cond, treatment_cond])].copy()
    subset["cond_bin"] = (subset["condition"] == treatment_cond).astype(int)
    if subset["cond_bin"].nunique() < 2:
        return None

    # scale fixation to [0,1] for numerical stability at higher powers
    subset["fx"] = subset["fixation_rate"].to_numpy(float) / 100.0

    # build formula: cond_bin * (fx + I(fx**2) + ... )
    poly_terms = " + ".join(
        ["fx"] + [f"I(fx**{j})" for j in range(2, degree + 1)]
    )
    formula = f"{dv} ~ cond_bin * ({poly_terms})"
    mod = smf.ols(formula, data=subset).fit()

    params = mod.params
    vcov   = mod.cov_params()
    df_resid = int(mod.df_resid)
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)

    # names of the simple-effect terms: cond_bin and cond_bin:<poly term>
    eff_names = ["cond_bin"]
    eff_names += [f"cond_bin:fx"] if degree >= 1 else []
    eff_names += [f"cond_bin:I(fx ** {j})" for j in range(2, degree + 1)]
    # guard against statsmodels naming variants
    eff_names = [n for n in eff_names if n in params.index]

    grid01 = FIX_RANGE / 100.0
    # contrast matrix C: rows = grid points, cols = eff terms
    # simple effect(x) = cond_bin + cond_bin:fx * x + cond_bin:fx^2 * x^2 + ...
    C = np.ones((len(grid01), len(eff_names)))
    for col, name in enumerate(eff_names):
        if name == "cond_bin":
            C[:, col] = 1.0
        elif name == "cond_bin:fx":
            C[:, col] = grid01
        else:
            # extract power j from I(fx ** j)
            j = int(name.split("**")[1].split(")")[0])
            C[:, col] = grid01 ** j

    beta = params[eff_names].to_numpy()
    Sig  = vcov.loc[eff_names, eff_names].to_numpy()

    effect = C @ beta
    var    = np.einsum("ij,jk,ik->i", C, Sig, C)
    var    = np.maximum(var, 1e-15)
    se     = np.sqrt(var)
    t_vals = effect / se

    sig_mask = np.abs(t_vals) >= t_crit
    # boundaries: grid points where significance flips
    flips = np.where(np.diff(sig_mask.astype(int)) != 0)[0]
    jn_points = sorted(float(FIX_RANGE[i + 1]) for i in flips)

    jn_result = dict(
        t_crit=t_crit,
        jn_points=jn_points,
        sig_mask=sig_mask,
        always_sig=bool(sig_mask.all()),
        never_sig=bool(not sig_mask.any()),
        region_label=_region_label(FIX_RANGE, sig_mask),
        t_curve=t_vals,
        se_curve=se,
        effect_curve=effect,
    )
    return effect, se, jn_result