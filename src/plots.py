"""
Publication-quality figures.

Design rules applied throughout:
  * categorical hues assigned in fixed order, never cycled or recoloured by rank
  * one y-axis per chart -- never a dual-axis chart
  * sequential encoding = one hue light->dark; diverging = blue/red with a neutral
    gray midpoint (zero is always the midpoint, and colour scales are symmetric)
  * legend present whenever >= 2 series, plus direct labels on <= 4 series so
    identity is never carried by colour alone
  * recessive grid and axes; thin marks; no chartjunk
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from . import config as C

# --- validated palette ------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e6e5e1"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

NEUTRAL_MID = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRAL_MID,
                 "#f3b0af", "#e34948", "#8f2020"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
)

PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def _style(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=12.5, fontweight="600",
                     loc="left", pad=12)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    return ax


def _save(fig, name: str):
    path = C.FIGURES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def _label_ends(ax, items, log: bool = False, min_gap_pts: float = 12.0):
    """
    Direct labels at the ends of lines, staggered vertically so they never collide.

    `items` is a list of (x, y, text, color). Satisfies the relief rule for
    low-contrast hues and removes reliance on the legend alone.
    """
    if not items:
        return
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    ys = [items[i][1] for i in order]
    span = (np.log10(max(ys)) - np.log10(min(ys))) if log else (max(ys) - min(ys))
    ax_h = ax.get_window_extent().height or 400
    lim = ax.get_ylim()
    full = (np.log10(lim[1]) - np.log10(lim[0])) if log else (lim[1] - lim[0])
    sep_pts = (span / full) * ax_h if full else 0

    offsets = [0.0] * len(items)
    if len(items) > 1 and sep_pts < min_gap_pts * (len(items) - 1):
        centre = (len(items) - 1) / 2
        for rank, i in enumerate(order):
            offsets[i] = (rank - centre) * min_gap_pts
    for i, (x, y, text, color) in enumerate(items):
        ax.annotate(text, xy=(x, y), xytext=(7, offsets[i]),
                    textcoords="offset points", color=color, fontsize=9,
                    fontweight="600", va="center")


# ---------------------------------------------------------------------------
# 1. Cumulative returns
# ---------------------------------------------------------------------------
def cumulative_returns(series: dict[str, pd.Series], name="01_cumulative_returns",
                       title="Growth of $1, net of costs (log scale)",
                       logy: bool = True):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    _style(ax, title, "Growth of $1")
    ends = []
    for i, (label, r) in enumerate(series.items()):
        eq = (1 + r.dropna()).cumprod()
        ax.plot(eq.index, eq.values, color=SERIES[i], linewidth=2.0,
                label=label, zorder=3)
        ends.append((eq.index[-1], eq.iloc[-1], f"{eq.iloc[-1]:.1f}x", SERIES[i]))
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left")
    ax.margins(x=0.08)
    ax.figure.canvas.draw()
    _label_ends(ax, ends, log=logy)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 2. Drawdowns
# ---------------------------------------------------------------------------
def drawdowns(series: dict[str, pd.Series], name="02_drawdowns",
              title="Drawdown from running peak"):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _style(ax, title, "Drawdown")
    for i, (label, r) in enumerate(series.items()):
        eq = (1 + r.dropna()).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, color=SERIES[i], linewidth=1.6, label=label)
        if i == 0:
            ax.fill_between(dd.index, dd.values, 0, color=SERIES[i], alpha=0.12)
    ax.yaxis.set_major_formatter(PCT)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower left")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 3. Rolling Sharpe
# ---------------------------------------------------------------------------
def rolling_sharpe(series: dict[str, pd.Series], window=252,
                   name="03_rolling_sharpe",
                   title="Rolling 12-month Sharpe ratio"):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _style(ax, title, "Sharpe (annualised)")
    for i, (label, r) in enumerate(series.items()):
        r = r.dropna()
        rs = np.sqrt(252) * r.rolling(window).mean() / r.rolling(window).std(ddof=1)
        ax.plot(rs.index, rs.values, color=SERIES[i], linewidth=1.6, label=label)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 4. Factor exposures through time
# ---------------------------------------------------------------------------
def factor_exposures(exposures: pd.DataFrame, name="04_factor_exposures",
                     title="Realised factor exposure of the long book (z-score units)"):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _style(ax, title, "Weighted-average z-score")
    sm = exposures.rolling(6, min_periods=1).mean()
    for i, col in enumerate(sm.columns):
        ax.plot(sm.index, sm[col].values, color=SERIES[i], linewidth=1.7,
                label=col.replace("_", " "))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper left")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 5. Turnover
# ---------------------------------------------------------------------------
def turnover_chart(turnover: pd.Series, name="05_turnover",
                   title="One-way turnover per monthly rebalance"):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    _style(ax, title, "Turnover (fraction of portfolio)")
    ax.bar(turnover.index, turnover.values, width=20, color=BLUE, alpha=0.75,
           linewidth=0)
    roll = turnover.rolling(12, min_periods=1).mean()
    ax.plot(roll.index, roll.values, color=ORANGE, linewidth=2.0,
            label="12-month average")
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper right")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 6. Annual returns
# ---------------------------------------------------------------------------
def annual_returns_bar(strat: pd.Series, bench: pd.Series,
                       labels=("Strategy", "SPY"),
                       name="06_annual_returns",
                       title="Calendar-year total return"):
    a = ((1 + strat.dropna()).resample("YE").prod() - 1)
    b = ((1 + bench.reindex(strat.index).fillna(0)).resample("YE").prod() - 1)
    years = a.index.year
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _style(ax, title, "Return")
    ax.bar(x - 0.21, a.values, 0.40, color=BLUE, label=labels[0], linewidth=0)
    ax.bar(x + 0.21, b.values, 0.40, color=ORANGE, label=labels[1], linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8.5)
    ax.yaxis.set_major_formatter(PCT)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 7. Monthly heatmap (diverging, symmetric about zero)
# ---------------------------------------------------------------------------
def monthly_heatmap(r: pd.Series, name="07_monthly_heatmap",
                    title="Monthly returns"):
    m = (1 + r.dropna()).resample("ME").prod() - 1
    df = pd.DataFrame({"y": m.index.year, "m": m.index.month, "v": m.values})
    piv = df.pivot(index="y", columns="m", values="v")
    lim = float(np.nanmax(np.abs(piv.values)))
    fig, ax = plt.subplots(figsize=(10.5, 0.30 * len(piv) + 2.0))
    _style(ax, title)
    im = ax.imshow(piv.values, cmap=DIVERGING, aspect="auto",
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8.5)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.yaxis.set_major_formatter(PCT)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 8. Factor IC
# ---------------------------------------------------------------------------
def factor_ic_bar(ic: pd.DataFrame, name="08_factor_ic",
                  title="Factor information coefficients, in-sample 1999-2012"):
    d = ic.drop(index="COMPOSITE", errors="ignore").sort_values("t_stat")
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    _style(ax, title, "IC t-statistic")
    colors = [BLUE if v > 0 else RED for v in d["t_stat"]]
    ax.barh(range(len(d)), d["t_stat"].values, color=colors, height=0.62, linewidth=0)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([i.replace("_", " ") for i in d.index], fontsize=9.5)
    for i, (v, ir) in enumerate(zip(d["t_stat"], d["mean_IC"])):
        ax.annotate(f"IC={ir:+.4f}", xy=(v, i), xytext=(6 if v > 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v > 0 else "right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    for t in (-1.96, 1.96):
        ax.axvline(t, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("t = ±1.96 (5% significance)", xy=(1.96, len(d) - 0.6),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.margins(x=0.22)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 9. Survivorship coverage
# ---------------------------------------------------------------------------
def survivorship_coverage(audit: pd.DataFrame, name="09_survivorship",
                          title="Data coverage: priced index members / true index members"):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    _style(ax, title, "Coverage")
    ax.plot(audit.index, audit["coverage"], color=BLUE, linewidth=2.0,
            label="Coverage achieved")
    ax.fill_between(audit.index, audit["coverage"], 1.0, color=RED, alpha=0.13,
                    label="Missing (unrecoverable survivorship hole)")
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower right")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 10. Null distribution
# ---------------------------------------------------------------------------
def null_histogram(null: pd.Series, actual: dict[str, float],
                   name="10_null_distribution", metric="Sharpe",
                   title="Random 50-stock portfolios vs the factor strategy"):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _style(ax, title, "Number of random portfolios", f"{metric} (annualised)")
    ax.hist(null.dropna(), bins=45, color="#9ec5f4", edgecolor=SURFACE, linewidth=0.5,
            label=f"Random portfolios (n={len(null.dropna())})")
    for i, (label, v) in enumerate(actual.items()):
        col = [ORANGE, AQUA, VIOLET][i % 3]
        ax.axvline(v, color=col, linewidth=2.2, label=f"{label} ({v:.3f})")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 11. Cost break-even
# ---------------------------------------------------------------------------
def cost_breakeven_chart(cb: pd.DataFrame, name="11_cost_breakeven",
                         title="Excess return over SPY vs one-way trading cost"):
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    _style(ax, title, "Excess CAGR vs SPY", "One-way trading cost (bps)")
    ax.plot(cb["total_bps_1way"], cb["excess_CAGR"], color=BLUE, linewidth=2.2,
            marker="o", markersize=6, label="Excess CAGR")
    ax.axhline(0, color=RED, linewidth=1.4, linestyle="--", label="Break-even")
    ax.fill_between(cb["total_bps_1way"], cb["excess_CAGR"], 0,
                    where=cb["excess_CAGR"] >= 0, color=BLUE, alpha=0.12)
    ax.fill_between(cb["total_bps_1way"], cb["excess_CAGR"], 0,
                    where=cb["excess_CAGR"] < 0, color=RED, alpha=0.12)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 12. Walk-forward folds
# ---------------------------------------------------------------------------
def walk_forward_chart(folds: pd.DataFrame, name="12_walk_forward",
                       title="Walk-forward: out-of-sample return by fold"):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _style(ax, title, "Annualised return", "Test window")
    x = np.arange(len(folds))
    ax.bar(x - 0.21, folds["test_CAGR"], 0.40, color=BLUE, label="Strategy", linewidth=0)
    if "bmk_CAGR" in folds:
        ax.bar(x + 0.21, folds["bmk_CAGR"], 0.40, color=ORANGE, label="SPY", linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([t.split("..")[0][:7] + "\n" + t.split("..")[1][:7]
                        for t in folds["test"]], fontsize=8)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 13. Rolling market beta
# ---------------------------------------------------------------------------
def rolling_beta_chart(rb: pd.DataFrame, name="13_rolling_beta",
                       title="Rolling 12-month factor betas"):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _style(ax, title, "Beta")
    cols = [c for c in rb.columns if c != "alpha"][:5]
    for i, c in enumerate(cols):
        ax.plot(rb.index, rb[c], color=SERIES[i], linewidth=1.6, label=c)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axhline(1, color=MUTED, linewidth=1, linestyle=":")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper left")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 14. In-sample vs out-of-sample
# ---------------------------------------------------------------------------
def is_oos_bar(data: pd.DataFrame, name="14_is_oos",
               title="In-sample vs out-of-sample: excess return over SPY"):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    _style(ax, title, "Excess CAGR vs SPY")
    x = np.arange(len(data.index))
    w = 0.38
    for i, col in enumerate(data.columns):
        ax.bar(x + (i - 0.5) * w, data[col].values, w, color=SERIES[i],
               label=col, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, fontsize=9.5)
    ax.axhline(0, color=INK2, linewidth=1.2)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2)
    return _save(fig, name)
