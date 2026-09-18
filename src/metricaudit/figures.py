"""P9: the figures, rendered from the results file rather than from the data.

Drawing from results rather than from frames means a figure cannot show a number
the results do not contain, which is the only way to keep a paper's figures and
its text describing the same run.

Each is written as a vector file for the publisher and a raster preview for
reading. Nothing here is committed to the repository: a figure is a finding and
findings travel with the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

INK = "#1f2a37"
ACCENT = "#b3441d"
MUTED = "#8a94a6"
FILL = "#dfe4ec"


def _save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.eps", format="eps")
    fig.savefig(directory / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  {stem}")


def _frame(results: dict, key: str) -> pd.DataFrame:
    return pd.DataFrame(results.get(key) or [])


def fig_reported_against_recomputed(results: dict, directory: Path) -> None:
    """Every indicator's reported value against what its source events support.

    On log axes with the line of equality drawn, so that a point's distance from
    the line reads as a ratio. Ratios are the right scale here because the
    divergences differ by orders of magnitude, and a linear axis would render
    every group except one indistinguishable from the origin.
    """
    per_group = _frame(results, "reconciliation_by_group")
    per_group = per_group[per_group["numeric"].astype(bool)]
    if per_group.empty:
        return

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    indicators = sorted(per_group["indicator"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for i, indicator in enumerate(indicators):
        block = per_group[per_group["indicator"] == indicator]
        x = block["recomputed"].astype(float).clip(lower=0.5)
        y = block["reported"].astype(float).clip(lower=0.5)
        diverges = ~block["matches"].astype(bool)
        ax.scatter(x[~diverges], y[~diverges], s=34, marker=markers[i % len(markers)],
                   facecolor="white", edgecolor=MUTED, linewidth=1.0, zorder=3)
        ax.scatter(x[diverges], y[diverges], s=42, marker=markers[i % len(markers)],
                   facecolor=ACCENT, edgecolor=ACCENT, alpha=0.85, zorder=4)

    lim = [0.5, max(per_group["reported"].astype(float).max(),
                    per_group["recomputed"].astype(float).max()) * 2.6]
    ax.plot(lim, lim, color=INK, linewidth=1.0, zorder=2)
    ax.text(lim[1] * 0.30, lim[1] * 0.30, "reported equals recomputed",
            rotation=40, fontsize=7.5, color=INK, ha="center", va="center",
            bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                      edgecolor="none"))
    for factor, label in ((2, "two times"), (10, "ten times")):
        ax.plot(lim, [v * factor for v in lim], color=MUTED, linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=1)
        ax.text(lim[1] * 0.45 / factor, lim[1] * 0.45, label, fontsize=7,
                color=MUTED, ha="center", va="center", rotation=40,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                          edgecolor="none"))

    extreme = per_group.loc[per_group["absolute_difference"].astype(float).idxmax()]
    ax.annotate(f"group {int(extreme['group_number'])}: "
                f"{int(extreme['reported'])} reported,\n"
                f"{int(extreme['recomputed'])} supported by source events",
                xy=(float(extreme["recomputed"]), float(extreme["reported"])),
                xytext=(float(extreme["recomputed"]) * 0.055,
                        float(extreme["reported"]) * 1.9),
                fontsize=7.5, color=ACCENT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.8,
                                shrinkA=3, shrinkB=5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("value recomputed from source events")
    ax.set_ylabel("value reported by the view")
    ax.set_aspect("equal")
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=ACCENT,
                      markeredgecolor=ACCENT, markersize=6, label="diverges"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                      markeredgecolor=MUTED, markersize=6, label="reconciles")]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8)
    _save(fig, directory, "figure_6_1_reported_against_recomputed")


def fig_divergence_ratio(results: dict, directory: Path) -> None:
    """The divergence ratio per group, with the extreme case marked."""
    per_group = _frame(results, "reconciliation_by_group")
    per_group = per_group[per_group["numeric"].astype(bool) & per_group["ratio"].notna()]
    divergent = sorted(set(
        _frame(results, "reconciliation_by_indicator")
        .query("classification == 'divergent'")["indicator"])
        & set(per_group["indicator"]))
    if not divergent:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    block = per_group[per_group["indicator"] == divergent[0]] \
        .sort_values("group_number")
    x = np.arange(len(block))
    ratios = block["ratio"].astype(float).to_numpy()
    colours = [ACCENT if not m else MUTED for m in block["matches"].astype(bool)]

    ax.bar(x, ratios, color=colours, width=0.68, zorder=3)
    ax.axhline(1.0, color=INK, linewidth=1.0, zorder=4)
    ax.annotate("no divergence", xy=(-0.5, 1.0), xytext=(-2, 4),
                textcoords="offset points", fontsize=7.5, color=INK,
                ha="right", va="bottom", annotation_clip=False)

    for xi, value in zip(x, ratios):
        ax.text(xi, value * 1.06, f"{value:.1f}", ha="center", va="bottom",
                fontsize=7, color=INK)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(g)}" for g in block["group_number"]])
    ax.set_xlabel("group")
    ax.set_ylabel("ratio of reported to recomputed")
    ax.set_ylim(0.8, ratios.max() * 2.6)
    ax.set_title(divergent[0].replace("_", " "), loc="left")
    _save(fig, directory, "figure_6_2_divergence_ratio")


def fig_agreement(results: dict, directory: Path) -> None:
    """M5: difference against mean on the log scale, with limits of agreement.

    The standard agreement plot, computed on logarithms so that the vertical
    axis reads as a ratio. A group at 0.3 is reported at twice what its events
    support; a group at 1.3 is reported at twenty times.
    """
    per_group = _frame(results, "reconciliation_by_group")
    per_group = per_group[per_group["numeric"].astype(bool)]
    consequence = results.get("consequence") or {}
    if per_group.empty or not consequence:
        return
    indicator = sorted(consequence)[0]
    block = per_group[per_group["indicator"] == indicator]
    rep = block["reported"].astype(float).to_numpy()
    rec = block["recomputed"].astype(float).to_numpy()
    keep = (rep > 0) & (rec > 0)
    if keep.sum() < 2:
        return
    diff = np.log10(rep[keep]) - np.log10(rec[keep])
    mean = (np.log10(rep[keep]) + np.log10(rec[keep])) / 2
    stats = consequence[indicator]["agreement"]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhspan(stats["lower"], stats["upper"], color=FILL, zorder=1)
    ax.axhline(stats["bias"], color=ACCENT, linewidth=1.2, zorder=3)
    ax.axhline(0.0, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.scatter(mean, diff, s=40, facecolor="white", edgecolor=INK, linewidth=1.1,
               zorder=4)

    for m, d, g in zip(mean, diff, block.loc[keep, "group_number"]):
        ax.annotate(f"{int(g)}", (m, d), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.5, color=MUTED)

    ax.margins(x=0.13)
    right = ax.get_xlim()[1]
    for y, colour, size, va, label in (
            (stats["bias"], ACCENT, 7.5, "bottom",
             f"mean {10 ** stats['bias']:.1f} times"),
            (stats["upper"], MUTED, 7, "bottom",
             f"upper limit of agreement, {10 ** stats['upper']:.1f} times"),
            (stats["lower"], MUTED, 7, "top",
             f"lower limit of agreement, {10 ** stats['lower']:.1f} times"),
            (0.0, INK, 7.5, "top", "reported equals recomputed")):
        ax.annotate(label, xy=(right, y),
                    xytext=(-3, 3 if va == "bottom" else -3),
                    textcoords="offset points", fontsize=size, color=colour,
                    ha="right", va=va)

    ax.set_xlabel("mean of reported and recomputed, on the base ten log scale")
    ax.set_ylabel("log difference, reported less recomputed")
    ax.set_title(indicator.replace("_", " "), loc="left")
    _save(fig, directory, "figure_6_3_agreement")


def fig_downstream_impact(results: dict, directory: Path) -> None:
    """What an analyst would have published, under each value set."""
    consequence = results.get("consequence") or {}
    if not consequence:
        return
    indicator = sorted(consequence)[0]
    imp = consequence[indicator]["impact"]

    labels = ("mean", "sd", "median", "max")
    reported = [imp["reported"][k] for k in labels]
    reconciled = [imp["reconciled"][k] for k in labels]
    reported_wo = [imp["reported_without_extreme"][k] for k in labels]
    reconciled_wo = [imp["reconciled_without_extreme"][k] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for ax, (rep, rec, title) in zip(axes, (
            (reported, reconciled, "all groups"),
            (reported_wo, reconciled_wo,
             f"excluding group {imp['extreme_group']}"))):
        x = np.arange(len(labels))
        ax.bar(x - 0.19, rep, width=0.36, color=ACCENT, label="reported", zorder=3)
        ax.bar(x + 0.19, rec, width=0.36, color=MUTED, label="reconciled", zorder=3)
        for xi, value in zip(x - 0.19, rep):
            ax.text(xi, value * 1.1, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color=ACCENT)
        for xi, value in zip(x + 0.19, rec):
            ax.text(xi, value * 1.1, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9)

    axes[0].set_ylim(top=max(reported + reconciled) * 6)
    axes[0].set_ylabel(indicator.replace("_", " "))
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, directory, "figure_6_4_downstream_impact")


def fig_mechanism_predictions(results: dict, directory: Path) -> None:
    """Each candidate mechanism against the observation it tries to explain.

    Reading down a column shows how far a mechanism was from the reported value
    for every group, which is the evidence that one candidate was selected and
    the rest were excluded rather than never considered.
    """
    matrix = _frame(results, "prediction_matrix")
    if matrix.empty:
        return
    indicator = matrix["indicator"].iloc[0]
    block = matrix[matrix["indicator"] == indicator].sort_values("group_number")
    candidates = [c for c in block.columns
                  if c not in ("indicator", "group_number", "group_id", "reported")]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(block))
    # The attributed candidate and the declared rule carry the argument, so they
    # get the ink. The rest need to be distinguishable from one another, or the
    # figure shows only that other candidates existed, not which one was where.
    spare = ["^", "v", "D", "P", "<", ">", "*"]
    for i, candidate in enumerate(c for c in candidates
                                  if c not in ("gauge_summation", "single_store")):
        ax.scatter(x, block[candidate].astype(float).clip(lower=0.5), s=26,
                   marker=spare[i % len(spare)], facecolor="none",
                   edgecolor=MUTED, linewidth=1.0, zorder=3,
                   label=candidate.replace("_", " "))
    for candidate, colour, marker, size, order in (
            ("single_store", INK, "s", 34, 4),
            ("gauge_summation", ACCENT, "o", 52, 5)):
        if candidate in candidates:
            ax.scatter(x, block[candidate].astype(float).clip(lower=0.5), s=size,
                       marker=marker, facecolor="none", edgecolor=colour,
                       linewidth=1.3, zorder=order,
                       label=candidate.replace("_", " "))
    ax.plot(x, block["reported"].astype(float).clip(lower=0.5), color=INK,
            linewidth=1.2, zorder=2, label="reported")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(g)}" for g in block["group_number"]])
    ax.set_xlabel("group")
    ax.set_ylabel(indicator.replace("_", " "))
    ax.margins(y=0.35)
    ax.set_ylim(top=block["reported"].astype(float).max() * 18)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left",
              columnspacing=1.0, handletextpad=0.3)
    _save(fig, directory, "figure_6_5_mechanism_predictions")


def fig_gauge_trajectory(results: dict, directory: Path) -> None:
    """The counter the view sums, in the order the snapshots were written.

    The attribution says the reported column is the sum of this series. The
    series is drawn so a reader can see what is being summed: a counter that
    climbs and then drops back to nothing, several times over, so that every
    step it takes is added again by each snapshot that follows it before the
    next drop. The shaded area is the sum, which is the reported figure.
    """
    trace = _frame(results, "gauge_trace")
    profile = _frame(results, "gauge_profile")
    if trace.empty:
        return

    group = int(trace["group_number"].iloc[0])
    row = profile[profile["group_number"] == group]
    x = trace["position"].astype(int).to_numpy()
    y = trace["counter"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.fill_between(x, 0, y, step="post", color=FILL, zorder=2)
    ax.step(x, y, where="post", color=INK, linewidth=1.3, zorder=3)

    for position in trace.loc[trace["reset_here"].astype(bool), "position"]:
        ax.axvline(int(position) - 0.5, color=ACCENT, linewidth=1.0,
                   linestyle=(0, (3, 2)), zorder=4)

    top = float(y.max())
    ax.set_ylim(0, top * 1.30)
    ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_xlabel("agent decision snapshots, in the order they were written")
    ax.set_ylabel("value of the counter")

    if not row.empty:
        summed = float(row["summed"].iloc[0])
        highest = float(row["highest_value"].iloc[0])
        ax.text(0.015, 0.97,
                f"shaded area, which the view reports: {summed:,.0f}\n"
                f"highest value the counter reaches: {highest:,.0f}",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                color=INK)

    # Label a reset in the clear stretch that follows one, rather than the
    # first, which sits under the summary text and among the densest steps.
    resets = [int(p) for p in trace.loc[trace["reset_here"].astype(bool), "position"]]
    if resets:
        marked = resets[len(resets) // 2]
        ax.annotate("the counter resets here", xy=(marked - 0.5, top * 0.90),
                    xytext=(marked - 4, top * 0.90),
                    fontsize=7.5, color=ACCENT, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.8,
                                    shrinkA=2, shrinkB=3))
    ax.grid(axis="y", color=FILL, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, directory, "figure_gauge_trajectory")


def fig_completeness(results: dict, directory: Path) -> None:
    """Completeness by check, with exact intervals."""
    frame = _frame(results, "completeness")
    if frame.empty:
        return
    frame = frame.sort_values("rate", ascending=True).reset_index(drop=True)
    labels = [f"{r['store']}: {r['check']}" for _, r in frame.iterrows()]

    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(frame) + 1.2))
    y = np.arange(len(frame))
    rates = frame["rate"].astype(float).to_numpy()
    lower = rates - frame["ci_lower"].astype(float).to_numpy()
    upper = frame["ci_upper"].astype(float).to_numpy() - rates

    ax.errorbar(rates, y, xerr=[lower, upper], fmt="o", color=INK,
                ecolor=MUTED, elinewidth=1.1, capsize=2.5, markersize=4.5,
                zorder=3)
    for yi, (_, row) in zip(y, frame.iterrows()):
        ax.text(float(row["ci_upper"]) + 0.006, yi,
                f"{int(row['n_affected'])}/{int(row['n_records'])}",
                fontsize=7, va="center", color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("proportion of records affected, with exact 95% interval")
    ax.set_xlim(left=-0.008)
    ax.margins(x=0.16)
    _save(fig, directory, "figure_6_6_completeness")


def fig_assumption_sweep(results: dict, directory: Path) -> None:
    """Which classifications survive a different analyst's assumptions."""
    sweep = _frame(results, "sweep")
    if sweep.empty:
        return
    pivot = sweep.pivot_table(index="indicator", columns="setting",
                              values="n_matching", aggfunc="first")
    # Alphabetical ordering interleaves the two assumption families, which reads
    # as though a rounding mode were an alternative to a deduplication mode.
    family = sweep.drop_duplicates("setting").set_index("setting")["assumption"]
    pivot = pivot[sorted(pivot.columns, key=lambda c: (family.get(c, ""), c))]
    total = sweep.groupby("indicator")["n_groups"].first()
    share = pivot.div(total, axis=0)

    fig, ax = plt.subplots(figsize=(6.8, 0.45 * len(share) + 1.6))
    data = share.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "agree", [ACCENT, "#e8d9b0", "#4f6f52"])
    cmap.set_bad("#f2f3f5")
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5,
                        color=MUTED)
            else:
                ax.text(j, i, f"{int(pivot.to_numpy()[i, j])}", ha="center",
                        va="center", fontsize=7.5,
                        color="white" if data[i, j] > 0.7 or data[i, j] < 0.25
                        else INK)

    ax.set_xticks(np.arange(share.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in share.columns], fontsize=7.5,
                       rotation=28, ha="right")
    ax.set_yticks(np.arange(share.shape[0]))
    ax.set_yticklabels([i.replace("_", " ") for i in share.index], fontsize=7.5)
    ax.set_xlabel("groups reconciling, of all groups, under each assumption",
                  labelpad=8)
    ax.tick_params(length=0)

    seen, boundary = [], []
    for j, column in enumerate(share.columns):
        name = family.get(column, "")
        if name not in seen:
            seen.append(name)
            if j:
                boundary.append(j)
        if name and (j == len(share.columns) - 1 or
                     family.get(share.columns[j + 1], "") != name):
            span = [k for k, c in enumerate(share.columns)
                    if family.get(c, "") == name]
            ax.text(np.mean(span), -0.70, name.replace("_", " "), ha="center",
                    va="bottom", fontsize=8.5, color=INK)
    for j in boundary:
        ax.axvline(j - 0.5, color="white", linewidth=2.4)

    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, directory, "figure_6_7_assumption_sweep")


def render_all(results_path: Path, directory: Path) -> None:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    print("rendering figures")
    for func in (fig_reported_against_recomputed, fig_divergence_ratio,
                 fig_agreement, fig_downstream_impact, fig_mechanism_predictions,
                 fig_gauge_trajectory, fig_completeness, fig_assumption_sweep):
        func(results, directory)
