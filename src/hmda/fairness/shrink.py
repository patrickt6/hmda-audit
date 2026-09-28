"""Empirical-Bayes lender watch list: which lenders' gaps survive small-sample noise.

The four-fifths screen (``hmda.fairness.air``) judges each lender on its own
raw rates. A lender with 12 Black applicants and 4 denials looks as extreme
as a lender with 12,000 and 4,000, although the first number is mostly
noise. This module ranks lenders by a *shrunken* gap instead.

Method (normal-normal empirical Bayes, stated plainly):

1. For each lender ``i`` with both groups present, the observed gap is
   ``g_i = denial_rate(group) - denial_rate(reference)``, in percentage points.
2. Its sampling variance is ``s_i^2 = p_g(1-p_g)/n_g + p_r(1-p_r)/n_r`` where
   ``p_g`` and ``p_r`` are the **pooled, all-lender** denial rates of the two
   groups, not the lender's own rates. A lender with 0 of 5 denials would
   otherwise get a variance of 0 and infinite weight.
3. The lenders' true gaps are modelled as draws from ``Normal(mu, tau^2)``.
   ``mu`` and ``tau^2`` are estimated by the DerSimonian-Laird method of
   moments, so there is no fitting loop and no tuning parameter.
4. Each lender's posterior is ``Normal(w_i g_i + (1-w_i) mu, w_i s_i^2)`` with
   ``w_i = tau^2 / (tau^2 + s_i^2)``. Small lenders (large ``s_i``) are pulled
   toward ``mu``; large lenders keep close to their own gap.
5. A lender is on the watch list when the posterior probability that its
   true gap exceeds the typical lender's gap ``mu`` is at least ``level``.

Limits: this is a *raw*-gap model. Raw and controlled
disparities always ship together, so a watch-list lender must
be reported next to its controlled gap from ``controlled.controlled_disparity``.
The normal approximation is poor for a group with very few applications;
``min_each`` sets a floor. No lender is ever named: lender
identities are for local review only, never for the site or README.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

#: Default posterior probability required to put a lender on the watch list.
DEFAULT_LEVEL = 0.95

#: Default minimum applications in EACH of the two groups for a lender to be scored.
DEFAULT_MIN_EACH = 5


@dataclass(frozen=True)
class LenderGap:
    lei: str
    n_group: int
    n_reference: int
    raw_gap_pp: float
    se_pp: float
    shrunk_gap_pp: float
    posterior_sd_pp: float
    weight: float
    p_above_typical: float
    on_watch_list: bool


@dataclass(frozen=True)
class ShrinkResult:
    group_value: str
    reference_group_value: str
    lenders_scored: int
    mu_pp: float
    tau_pp: float
    pooled_rate_group: float
    pooled_rate_reference: float
    level: float
    lenders: tuple[LenderGap, ...]

    @property
    def watch_list(self) -> tuple[LenderGap, ...]:
        return tuple(l for l in self.lenders if l.on_watch_list)


def _normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal Z."""
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def shrink_lender_gaps(
    rows,
    group_value: str,
    reference_group_value: str,
    *,
    min_each: int = DEFAULT_MIN_EACH,
    level: float = DEFAULT_LEVEL,
) -> ShrinkResult:
    """Score every lender's gap between ``group_value`` and ``reference_group_value``.

    ``rows`` are dicts with keys ``lei``, ``group_value``, ``denominator`` and
    ``denials``, exactly the rows ``sql/air_by_lei_group.sql`` returns.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")

    counts: dict[str, dict[str, tuple[int, int]]] = {}
    for r in rows:
        if r["group_value"] in (group_value, reference_group_value):
            counts.setdefault(r["lei"], {})[r["group_value"]] = (int(r["denominator"]), int(r["denials"]))

    both = {
        lei: c
        for lei, c in counts.items()
        if group_value in c and reference_group_value in c
        and c[group_value][0] >= min_each and c[reference_group_value][0] >= min_each
    }
    if len(both) < 3:
        raise ValueError(f"only {len(both)} lender(s) have both groups; too few to estimate the spread")

    n_g = sum(c[group_value][0] for c in both.values())
    d_g = sum(c[group_value][1] for c in both.values())
    n_r = sum(c[reference_group_value][0] for c in both.values())
    d_r = sum(c[reference_group_value][1] for c in both.values())
    p_g, p_r = d_g / n_g, d_r / n_r
    if p_g * (1 - p_g) + p_r * (1 - p_r) == 0:
        # Both pooled rates are 0 or 1, so every lender's variance is 0 and the
        # moment estimate divides by zero. There is no spread to estimate.
        raise ValueError("pooled denial rates are 0 or 1 in both groups; the gaps have no sampling variance")

    leis = sorted(both)
    gaps, variances = [], []
    for lei in leis:
        (ng, dg), (nr, dr) = both[lei][group_value], both[lei][reference_group_value]
        gaps.append(100.0 * (dg / ng - dr / nr))
        variances.append(1e4 * (p_g * (1 - p_g) / ng + p_r * (1 - p_r) / nr))

    # DerSimonian-Laird moment estimate of mu and tau^2.
    w = [1.0 / v for v in variances]
    sw = sum(w)
    mu_fixed = sum(wi * g for wi, g in zip(w, gaps)) / sw
    q = sum(wi * (g - mu_fixed) ** 2 for wi, g in zip(w, gaps))
    c = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - (len(gaps) - 1)) / c)
    w_re = [1.0 / (v + tau2) for v in variances]
    mu = sum(wi * g for wi, g in zip(w_re, gaps)) / sum(w_re)

    lenders = []
    for lei, g, v in zip(leis, gaps, variances):
        shrink_w = tau2 / (tau2 + v) if tau2 > 0 else 0.0
        post_mean = shrink_w * g + (1 - shrink_w) * mu
        post_sd = sqrt(shrink_w * v)
        p_above = _normal_sf((mu - post_mean) / post_sd) if post_sd > 0 else 0.0
        (ng, _), (nr, _) = both[lei][group_value], both[lei][reference_group_value]
        lenders.append(LenderGap(
            lei=lei, n_group=ng, n_reference=nr, raw_gap_pp=g, se_pp=sqrt(v),
            shrunk_gap_pp=post_mean, posterior_sd_pp=post_sd, weight=shrink_w,
            p_above_typical=p_above, on_watch_list=p_above >= level,
        ))

    lenders.sort(key=lambda l: l.p_above_typical, reverse=True)
    return ShrinkResult(
        group_value=group_value, reference_group_value=reference_group_value,
        lenders_scored=len(lenders), mu_pp=mu, tau_pp=sqrt(tau2),
        pooled_rate_group=p_g, pooled_rate_reference=p_r, level=level,
        lenders=tuple(lenders),
    )
