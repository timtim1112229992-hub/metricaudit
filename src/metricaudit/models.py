"""The estimators, kept apart from the procedures that call them.

Twelve groups is a small number, and the arithmetic here is chosen for that.
Exact intervals are used wherever a quantity is a proportion, resampling is
stratified over groups rather than over records because groups are the
independent unit, and no procedure is offered that would need an asymptotic
argument to be trusted at this size.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def clopper_pearson(successes: int, trials: int, conf: float = 0.95) -> tuple[float, float]:
    """M8: the exact interval for a proportion.

    Exact rather than normal-approximate because the proportions of interest
    here are near zero and near one, which is precisely where the approximation
    returns bounds outside the unit interval.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - conf
    lower = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes,
                                                      trials - successes + 1)
    upper = 1.0 if successes == trials else stats.beta.ppf(
        1 - alpha / 2, successes + 1, trials - successes)
    return (float(lower), float(upper))


def exact_upper_bound(trials: int, conf: float = 0.95) -> float:
    """The one-sided upper bound on a rate whose observed count is zero.

    Returned so that an absence carries a stated limit instead of a bare zero.
    This is the exact bound, 1 - (1 - conf) ** (1 / n), not the familiar 3 / n
    approximation to it; at these sizes the approximation is anti-conservative
    and the two differ visibly.
    """
    if trials <= 0:
        return float("nan")
    return float(1.0 - (1.0 - conf) ** (1.0 / trials))


def _percentile_of(values: np.ndarray, point: float) -> float:
    return float(np.mean(values < point))


def bca_interval(sample: np.ndarray, statistic, n_boot: int, seed: int,
                 conf: float = 0.95) -> dict:
    """M4: a bias-corrected and accelerated bootstrap interval over groups.

    Bias correction matters here because the quantities being resampled are
    ratios over a dozen units and their sampling distributions are markedly
    skewed. A percentile interval on a skewed statistic is centred in the wrong
    place, and with twelve units there is no appeal to symmetry available.

    Returns the estimate, the interval, and the flag saying whether acceleration
    could be computed, since a degenerate jackknife is silent otherwise.
    """
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    n = len(sample)
    observed = float(statistic(sample)) if n else float("nan")
    if n < 3:
        return {"estimate": observed, "lower": float("nan"), "upper": float("nan"),
                "n_units": n, "method": "not estimable below three units",
                "accelerated": False}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    replicates = np.array([statistic(sample[idx]) for idx in draws], dtype=float)
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size < n_boot // 2:
        return {"estimate": observed, "lower": float("nan"), "upper": float("nan"),
                "n_units": n, "method": "too many non-finite replicates",
                "accelerated": False}

    prop = _percentile_of(replicates, observed)
    if prop <= 0 or prop >= 1:
        # Every replicate fell on one side of the estimate, so the normal
        # quantile of the proportion is infinite and the correction cannot be
        # formed. Falling back to the percentile interval is honest; silently
        # clipping the proportion would manufacture a correction.
        alpha = 1 - conf
        return {"estimate": observed,
                "lower": float(np.quantile(replicates, alpha / 2)),
                "upper": float(np.quantile(replicates, 1 - alpha / 2)),
                "n_units": n, "method": "percentile; bias correction undefined",
                "accelerated": False}

    z0 = stats.norm.ppf(prop)

    jack = np.array([statistic(np.delete(sample, i)) for i in range(n)], dtype=float)
    jack_mean = jack.mean()
    diff = jack_mean - jack
    denom = 6.0 * (np.sum(diff ** 2) ** 1.5)
    acc = float(np.sum(diff ** 3) / denom) if denom > 0 else 0.0

    alpha = 1 - conf
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)

    def adjust(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))

    lo, hi = adjust(z_lo), adjust(z_hi)
    return {"estimate": observed,
            "lower": float(np.quantile(replicates, np.clip(lo, 0, 1))),
            "upper": float(np.quantile(replicates, np.clip(hi, 0, 1))),
            "n_units": n, "method": "bias-corrected and accelerated",
            "accelerated": denom > 0}


def limits_of_agreement(difference: np.ndarray, conf: float = 0.95) -> dict:
    """M5: the summary that accompanies a difference-against-mean plot.

    Reported on the log scale, where the differences are ratios and the
    assumption of roughly constant spread is defensible. On the raw scale a
    single unbounded counter sets the width of the limits by itself, which
    describes that counter rather than the agreement.
    """
    diff = np.asarray(difference, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return {"bias": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "n": int(diff.size)}
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    z = float(stats.norm.ppf(1 - (1 - conf) / 2))
    return {"bias": bias, "sd": sd, "lower": bias - z * sd, "upper": bias + z * sd,
            "n": int(diff.size)}


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """M10: the representative association, estimated the same way twice.

    Rank-based, because the point of estimating it twice is to see how far a
    substantive conclusion moves when one variable is replaced, and a
    product-moment coefficient on a variable containing an unbounded counter
    would move for reasons that have nothing to do with the substance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"rho": float("nan"), "p": float("nan"), "n": int(x.size),
                "estimable": False}
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(x.size), "estimable": True}


def rank_order(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(np.asarray(values, dtype=float), method="average")


def kendall_tau(a: np.ndarray, b: np.ndarray) -> dict:
    """How far a ranking moves between two value sets.

    Used for the consequence of the fault rather than for a substantive claim:
    if reported and reconciled values rank the groups differently, then any
    analysis that ordered groups by the reported figure ordered them wrongly.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return {"tau": float("nan"), "p": float("nan"), "n": int(a.size),
                "estimable": False}
    tau, p = stats.kendalltau(a, b)
    return {"tau": float(tau), "p": float(p), "n": int(a.size), "estimable": True}
