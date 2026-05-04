"""
DCA Holy Grail — Figure generation.

Produces 4 figures saved to figures/dca/:
  fig1_equity_curves_dca.png   Multi-strategy equity curves + 4 crisis zooms.
  fig2_dual_drawdown.png       Dual-panel DD: vs market value + vs cumulative
                               contributions (the DCA-specific lens).
  fig3_r_cross_section.png     Metric vs r-bucket cross-section.
  fig4_cpcv_dca_distribution.png  CPCV rolling-window XIRR distributions
                                  (only if results/dca/cpcv_dca_results.pkl
                                  exists; otherwise gracefully skipped).

Reuses (does NOT modify) scripts.dca.core and scripts.dca.retail_holy_grail.
If results/dca/baselines_results.pkl is already on disk it is loaded as a
cache to avoid re-running the slow yfinance download + backtests.

Run:
    python -m scripts.dca.figures_dca
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import load_universe
from .retail_holy_grail import cross_section


# --------------------------------------------------------------------------
# Paths and styling
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "figures" / "dca"
RESULTS_DIR = REPO_ROOT / "results" / "dca"
BASELINES_PKL = RESULTS_DIR / "baselines_results.pkl"
CPCV_PKL = RESULTS_DIR / "cpcv_dca_results.pkl"

# Print-friendly palette (B/W distinguishable via line style + marker)
COLORS = {
    "Retail Holy Grail":             "#b22222",   # firebrick
    "DCA QQQ B&H":                   "#1f4e79",   # navy
    "DCA SPY B&H":                   "#5b9bd5",   # mid-blue
    "DCA TQQQ B&H":                  "#7030a0",   # purple
    "DCA 60/40 (QQQ + AGG)":         "#2e7d32",   # green
    "DCA 60/40 (SPY + AGG)":         "#388e3c",
    "Pure DCA Rotation (no r-rule)": "#ed7d31",   # orange
    "Lump-sum Rotation":             "#000000",   # black
}
LINESTYLES = {
    "Retail Holy Grail":             "-",
    "DCA QQQ B&H":                   "--",
    "DCA SPY B&H":                   ":",
    "DCA TQQQ B&H":                  "-.",
    "DCA 60/40 (QQQ + AGG)":         (0, (3, 1, 1, 1)),
    "DCA 60/40 (SPY + AGG)":         (0, (1, 1)),
    "Pure DCA Rotation (no r-rule)": (0, (5, 2)),
    "Lump-sum Rotation":             "-",
}

# Crises zoom windows (start, end, label)
CRISES = [
    ("2000-01-01", "2003-12-31", "Dot-com Unwind"),
    ("2007-06-01", "2010-06-30", "Global Financial Crisis"),
    ("2020-01-01", "2020-12-31", "COVID Shock"),
    ("2022-01-01", "2023-06-30", "AI Bull + 2022 Selloff"),
]


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Result builder (cache-aware)
# --------------------------------------------------------------------------

def _build_or_load_baselines(U: dict, initial: float, annual: float,
                             contrib_freq: str = "M") -> dict:
    """Try to load cached baselines pickle; otherwise rebuild via baselines.py.

    The pickle is regarded as valid only when its (initial, annual, freq) matches
    the request. Otherwise we recompute.
    """
    if BASELINES_PKL.exists():
        try:
            with open(BASELINES_PKL, "rb") as f:
                data = pickle.load(f)
            args_ok = (data.get("args", {}).get("initial") == initial
                       and data.get("args", {}).get("annual") == annual
                       and data.get("args", {}).get("freq") == contrib_freq)
            if args_ok and "results" in data:
                print(f"[figures_dca] Loaded cached baselines from {BASELINES_PKL}")
                return data["results"]
            print("[figures_dca] Cache parameters mismatch; rebuilding baselines.")
        except Exception as e:
            print(f"[figures_dca] Cache load failed ({e}); rebuilding baselines.")

    # Lazy import to avoid circulars when this module is reused via retail_holy_grail
    from .baselines import run_baselines
    print(f"[figures_dca] Running baselines (initial=${initial:,.0f}, "
          f"annual=${annual:,.0f}, freq={contrib_freq}) ...")
    return run_baselines(initial, annual, U, contrib_freq=contrib_freq)


# --------------------------------------------------------------------------
# Figure 1: Equity curves with crisis zooms
# --------------------------------------------------------------------------

def fig1_equity_curves(U: dict, initial: float = 10_000,
                       annual: float = 10_000) -> plt.Figure:
    """Multi-strategy equity curves: full 27-year main panel + 4 crisis zooms."""
    results = _build_or_load_baselines(U, initial, annual, "M")

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.6, 1.0, 1.0],
                           hspace=0.55, wspace=0.40,
                           left=0.07, right=0.96, top=0.94, bottom=0.06)

    ax_main = fig.add_subplot(gs[0, :])

    # Main panel: full history, log scale
    plot_order = [
        "Retail Holy Grail",
        "DCA QQQ B&H",
        "DCA SPY B&H",
        "DCA TQQQ B&H",
        "DCA 60/40 (QQQ + AGG)",
        "Lump-sum Rotation",
    ]
    legend_handles = []
    for name in plot_order:
        if name not in results:
            continue
        eq = results[name]["result"].equity
        m = results[name]["metrics"]
        line, = ax_main.plot(eq.index, eq.values,
                              color=COLORS.get(name, "#444"),
                              linestyle=LINESTYLES.get(name, "-"),
                              linewidth=1.6,
                              label=(f"{name}: ${m['final_value']:,.0f} "
                                     f"| XIRR {m['xirr']*100:.2f}% "
                                     f"| MDD {m['mdd_market']*100:.1f}%"))
        legend_handles.append(line)

    # Cumulative contributions reference line (a 45° diagonal, very useful
    # because everything *below* this line means MV < $ contributed)
    if "Retail Holy Grail" in results:
        cc = results["Retail Holy Grail"]["result"].contributions_cum
        # Replace zeros with NaN so log-scale is happy
        cc_plot = cc.replace(0, np.nan)
        ref, = ax_main.plot(cc_plot.index, cc_plot.values,
                             color="grey", linestyle="-.",
                             linewidth=1.2, alpha=0.85,
                             label=f"Cumulative contributions "
                                   f"(end ${cc.iloc[-1]:,.0f})")
        legend_handles.append(ref)

    ax_main.set_yscale("log")
    ax_main.set_ylabel("Portfolio value ($, log scale)", fontsize=11)
    ax_main.set_title(f"Retail DCA Holy Grail vs Baselines — "
                       f"${initial:,.0f} initial + ${annual:,.0f}/yr "
                       f"(monthly DCA, 1999-2026)",
                       fontsize=13, fontweight="bold")
    ax_main.grid(True, which="both", linestyle=":", alpha=0.4)
    # Legend in upper-left corner where curves are still small (early years)
    leg = ax_main.legend(handles=legend_handles, fontsize=8.5,
                          loc="upper left", frameon=True, framealpha=0.95,
                          ncol=1)
    leg.get_frame().set_edgecolor("grey")
    ax_main.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_main.set_xlim(pd.Timestamp("1999-01-01"), pd.Timestamp("2026-06-01"))

    # Annotate major regime events with vertical lines + labels near the top
    events = [
        ("2000-03-10", "Dot-com peak"),
        ("2008-09-15", "Lehman"),
        ("2020-03-23", "COVID low"),
        ("2022-10-13", "2022 trough"),
    ]
    ymin, ymax = ax_main.get_ylim()
    label_y = ymax * 0.45  # above mid-curves, below legend
    for date, lbl in events:
        ts = pd.Timestamp(date)
        if ts in results["Retail Holy Grail"]["result"].equity.index:
            ax_main.axvline(ts, color="grey", linestyle=":",
                             linewidth=0.8, alpha=0.55)
            ax_main.text(ts, label_y, " " + lbl, rotation=90,
                          fontsize=8, color="#444",
                          va="center", ha="left",
                          bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                     ec="none", alpha=0.75))

    # Zoom panels (rows 1 and 2)
    zoom_strategies = ["Retail Holy Grail", "DCA QQQ B&H", "Lump-sum Rotation"]
    for i, (start, end, label) in enumerate(CRISES):
        row = 1 + i // 2
        col_start = (i % 2) * 2
        ax = fig.add_subplot(gs[row, col_start:col_start + 2])
        ts0, ts1 = pd.Timestamp(start), pd.Timestamp(end)
        for name in zoom_strategies:
            if name not in results:
                continue
            eq = results[name]["result"].equity
            mask = (eq.index >= ts0) & (eq.index <= ts1)
            sl = eq[mask]
            if sl.empty:
                continue
            ax.plot(sl.index, sl.values,
                    color=COLORS.get(name, "#444"),
                    linestyle=LINESTYLES.get(name, "-"),
                    linewidth=1.5,
                    label=name)
        # Endpoint summary as a single text block in upper-left of zoom panel
        info_lines = []
        for name in zoom_strategies:
            if name not in results:
                continue
            eq = results[name]["result"].equity
            mask = (eq.index >= ts0) & (eq.index <= ts1)
            sl = eq[mask]
            if sl.empty:
                continue
            chg = sl.iloc[-1] / sl.iloc[0] - 1.0 if sl.iloc[0] > 0 else 0.0
            info_lines.append((name, sl.iloc[-1], chg))
        if info_lines:
            txt = "\n".join(
                f"{n}: ${v/1000:.1f}K ({c*100:+.0f}%)"
                for n, v, c in info_lines)
            ax.text(0.02, 0.97, txt, transform=ax.transAxes,
                    fontsize=7.5, va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="grey", alpha=0.92, lw=0.5))
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(axis="x", labelsize=8, rotation=0)
        ax.tick_params(axis="y", labelsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        # Format y-axis as $K
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K" if v >= 1000
                              else f"${v:.0f}"))
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Figure 1 — DCA Holy Grail Equity Curves with Crisis Zooms",
                 fontsize=14, fontweight="bold", y=0.995)

    out = FIG_DIR / "fig1_equity_curves_dca.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[figures_dca] Saved {out}")
    return fig


# --------------------------------------------------------------------------
# Figure 2: Dual-panel drawdown
# --------------------------------------------------------------------------

def fig2_dual_drawdown(U: dict, initial: float = 10_000,
                       annual: float = 10_000) -> plt.Figure:
    """Two stacked panels: DD vs market value + DD vs cumulative contributions.

    Panel B is the DCA-specific lens — it shows the *negative* excursions of
    (equity − contributions) / contributions. We deliberately clip out the
    positive (gain) territory because the natural growth dwarfs the underwater
    excursions on a DCA portfolio. Annotated maximum-underwater trough per
    strategy.
    """
    results = _build_or_load_baselines(U, initial, annual, "M")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1],
                                                  "hspace": 0.22,
                                                  "left": 0.08,
                                                  "right": 0.96,
                                                  "top": 0.93,
                                                  "bottom": 0.07})

    show = ["Retail Holy Grail", "DCA QQQ B&H", "Lump-sum Rotation"]

    worst_records: list[dict] = []
    for name in show:
        if name not in results:
            continue
        res = results[name]["result"]
        eq = res.equity
        cc = res.contributions_cum
        col = COLORS.get(name, "#444")
        ls = LINESTYLES.get(name, "-")

        # Panel 1: market-value drawdown (traditional)
        roll_max = eq.cummax()
        dd_mkt = (eq / roll_max - 1.0) * 100  # percent
        m = results[name]["metrics"]
        ax1.fill_between(dd_mkt.index, dd_mkt.values, 0,
                          color=col, alpha=0.16)
        ax1.plot(dd_mkt.index, dd_mkt.values,
                 color=col, linestyle=ls, linewidth=1.4,
                 label=f"{name} (worst {m['mdd_market']*100:.1f}%)")

        # Panel 2: underwater-vs-contributions only (clip positive territory)
        common = eq.index.intersection(cc.index)
        eq_c = eq.loc[common]
        cc_c = cc.loc[common]
        mask = cc_c > 0
        ratio = pd.Series(np.nan, index=common)
        ratio.loc[mask] = (eq_c[mask] - cc_c[mask]) / cc_c[mask] * 100
        # Clip to underwater region: anything > 0 is rendered as 0
        underwater = ratio.clip(upper=0)
        ax2.fill_between(underwater.index, underwater.values, 0,
                          where=~underwater.isna(), color=col, alpha=0.18)
        ax2.plot(underwater.index, underwater.values,
                 color=col, linestyle=ls, linewidth=1.4,
                 label=f"{name} (worst {m['mdd_vs_contrib']*100:.1f}%)")

        # Track the worst trough on the contribution-relative series
        if mask.any() and ratio.min() < 0:
            i_worst = ratio.idxmin()
            worst_records.append({
                "name": name,
                "color": col,
                "date": i_worst,
                "equity": float(eq.loc[i_worst]),
                "contributed": float(cc.loc[i_worst]),
                "ratio_pct": float(ratio.loc[i_worst]),
                "dd_mkt_pct": float(dd_mkt.loc[i_worst]),
            })

    # Panel 1 styling
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Drawdown vs market peak (%)", fontsize=10)
    ax1.set_title("Panel A — Market-value drawdown (traditional lens)",
                   fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92)
    ax1.set_ylim(top=5)

    # Panel 2 styling
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("(Equity − Contrib) / Contrib (%, underwater only)",
                    fontsize=10)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_title("Panel B — Underwater vs cumulative contributions "
                   "(retail-DCA lens: \"how far is MV below the cash I put in?\")",
                   fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92)
    ax2.set_ylim(top=5)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Mark worst-trough with markers + a single tabular text-box (right side)
    if worst_records:
        # Sort by date so stagger is left-to-right
        worst_records.sort(key=lambda r: r["date"])
        for rec in worst_records:
            ax2.plot(rec["date"], rec["ratio_pct"],
                     marker="v", color=rec["color"],
                     markersize=10, markeredgecolor="white",
                     markeredgewidth=1.0, zorder=10)
        # Build a single tabular text block summarizing each trough
        header = f"{'Strategy':<22}{'Date':>11}{'Contrib':>12}{'MV':>12}{'DD-contr':>11}{'DD-mkt':>10}"
        rows = [header]
        for rec in worst_records:
            rows.append(
                f"{rec['name'][:22]:<22}"
                f"{str(rec['date'].date()):>11}"
                f"  ${rec['contributed']:>9,.0f}"
                f"  ${rec['equity']:>9,.0f}"
                f"  {rec['ratio_pct']:>7.1f}%"
                f"  {rec['dd_mkt_pct']:>6.1f}%"
            )
        block = "\n".join(rows)
        ax2.text(0.99, 0.04, block, transform=ax2.transAxes,
                  fontsize=7.5, family="monospace",
                  va="bottom", ha="right",
                  bbox=dict(boxstyle="round,pad=0.4",
                            fc="white", ec="grey", lw=0.6, alpha=0.95))

    fig.suptitle(f"Figure 2 — Dual-panel Drawdown "
                 f"(${initial:,.0f} + ${annual:,.0f}/yr DCA, 1999-2026)",
                 fontsize=13, fontweight="bold", y=0.995)

    out = FIG_DIR / "fig2_dual_drawdown.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[figures_dca] Saved {out}")
    return fig


# --------------------------------------------------------------------------
# Figure 3: r-bucket cross-section
# --------------------------------------------------------------------------

def fig3_r_cross_section(U: dict) -> plt.Figure:
    """Run Retail Holy Grail across many r-bucket profiles and chart metrics."""
    # r ∈ {0.5, 1, 2, 3, 5, 10, inf}
    profiles = [
        (5_000, 10_000),     # r=0.5
        (10_000, 10_000),    # r=1.0
        (10_000, 5_000),     # r=2.0
        (30_000, 10_000),    # r=3.0
        (50_000, 10_000),    # r=5.0
        (100_000, 10_000),   # r=10
        (10_000, 0),         # r=inf
    ]
    print("[figures_dca] Running r-bucket cross-section ...")
    df = cross_section(profiles, U)
    df = df.sort_values("r").reset_index(drop=True)

    # Use a string label for x-axis (inf is hard to plot numerically)
    df["r_label"] = df["r"].apply(lambda r: "inf" if not np.isfinite(r) else f"{r:.1f}")
    df["xpos"] = np.arange(len(df))
    # Compact profile string for x-tick under panel C
    def _profile_str(init, ann):
        if ann <= 0:
            return f"${init:,.0f}\n+ $0/yr"
        return f"${init:,.0f}\n+ ${ann:,.0f}/yr"
    df["profile_str"] = [_profile_str(i, a)
                          for i, a in zip(df["initial"], df["annual"])]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12.5),
                              gridspec_kw={"hspace": 0.55,
                                            "left": 0.08,
                                            "right": 0.96,
                                            "top": 0.95,
                                            "bottom": 0.08})

    # Panel A: XIRR + TWR
    ax = axes[0]
    bw = 0.36
    ax.bar(df["xpos"] - bw / 2, df["xirr"] * 100, width=bw,
           color="#b22222", edgecolor="black", linewidth=0.6,
           label="XIRR (money-weighted)")
    ax.bar(df["xpos"] + bw / 2, df["twr"] * 100, width=bw,
           color="#1f4e79", edgecolor="black", linewidth=0.6,
           label="TWR (time-weighted, signal-only)")
    for _, row in df.iterrows():
        ax.text(row["xpos"] - bw / 2, row["xirr"] * 100 + 0.4,
                f"{row['xirr']*100:.1f}", ha="center", fontsize=8)
        ax.text(row["xpos"] + bw / 2, row["twr"] * 100 + 0.4,
                f"{row['twr']*100:.1f}", ha="center", fontsize=8)
    ax.set_xticks(df["xpos"])
    ax.set_xticklabels([f"r={r}" for r in df["r_label"]], fontsize=9)
    ax.set_ylabel("Annualised return (%)", fontsize=10)
    ax.set_title("Panel A — Returns vs r-bucket: "
                  "XIRR (money-weighted) vs TWR (time-weighted)",
                  fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(top=max(df["xirr"].max(), df["twr"].max()) * 100 + 5)
    _annotate_r_boundaries(ax, df)

    # Panel B: drawdowns
    ax = axes[1]
    ax.bar(df["xpos"] - bw / 2, df["mdd_market"] * 100, width=bw,
           color="#5b9bd5", edgecolor="black", linewidth=0.6,
           label="MDD vs market peak")
    ax.bar(df["xpos"] + bw / 2, df["mdd_vs_contrib"] * 100, width=bw,
           color="#ed7d31", edgecolor="black", linewidth=0.6,
           label="MDD vs cumulative contributions")
    for _, row in df.iterrows():
        ax.text(row["xpos"] - bw / 2, row["mdd_market"] * 100 - 3,
                f"{row['mdd_market']*100:.1f}", ha="center", fontsize=8,
                color="#0d3a6b")
        ax.text(row["xpos"] + bw / 2, row["mdd_vs_contrib"] * 100 - 3,
                f"{row['mdd_vs_contrib']*100:.1f}", ha="center", fontsize=8,
                color="#a04000")
    ax.set_xticks(df["xpos"])
    ax.set_xticklabels([f"r={r}" for r in df["r_label"]], fontsize=9)
    ax.set_ylabel("Drawdown (%, more negative = worse)", fontsize=10)
    ax.set_title("Panel B — Drawdowns vs r-bucket: "
                  "market peak (traditional) vs cumulative contributions (DCA-lens)",
                  fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(bottom=min(df["mdd_market"].min(),
                            df["mdd_vs_contrib"].min()) * 100 - 12,
                top=8)
    _annotate_r_boundaries(ax, df)

    # Panel C: multiple — show profile under each bar
    ax = axes[2]
    bars = ax.bar(df["xpos"], df["multiple"], width=0.55,
                   color="#7030a0", edgecolor="black", linewidth=0.6)
    for rect, mult in zip(bars, df["multiple"]):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(df["multiple"]) * 0.012,
                f"{mult:.2f}x",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(df["xpos"])
    # Two-line tick labels: "r=X.X" then "$init + $ann/yr"
    tick_labels = [f"r={r}\n{p}" for r, p in zip(df["r_label"], df["profile_str"])]
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_xlabel("Profile", fontsize=10)
    ax.set_ylabel("Final / Total contributed (multiple)", fontsize=10)
    ax.set_title("Panel C — Wealth multiple (final value / total contributed)\n"
                  "Note: lower r → bigger multiple, because the signal had longer "
                  "to compound the *initial* lump",
                  fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax.set_ylim(top=max(df["multiple"]) * 1.12)
    _annotate_r_boundaries(ax, df)

    fig.suptitle("Figure 3 — Retail Holy Grail Cross-section by r-bucket",
                 fontsize=13, fontweight="bold", y=0.995)

    out = FIG_DIR / "fig3_r_cross_section.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[figures_dca] Saved {out}")
    return fig


def _annotate_r_boundaries(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Vertical lines at r=2 and r=5 — the LSI bucket switches per the r-rule."""
    boundaries = [(2.0, "r=2: 100% LSI ↔ 70% LSI", "#444"),
                  (5.0, "r=5: 70% LSI ↔ 50% LSI", "#888")]
    for r_val, lbl, c in boundaries:
        # Find x-position between bars at this r value
        finite = df[np.isfinite(df["r"])]
        if r_val < finite["r"].min() or r_val > finite["r"].max():
            continue
        # If r_val matches a bar exactly, place line on its right edge
        if (finite["r"] == r_val).any():
            xp = float(finite[finite["r"] == r_val]["xpos"].iloc[0]) + 0.5
        else:
            below = finite[finite["r"] < r_val]
            above = finite[finite["r"] > r_val]
            if below.empty or above.empty:
                continue
            xp = (float(below["xpos"].iloc[-1])
                  + float(above["xpos"].iloc[0])) / 2.0
        ax.axvline(xp, color=c, linestyle="--", linewidth=1.0, alpha=0.7)
        # Label
        ymin, ymax = ax.get_ylim()
        ax.text(xp, ymax - (ymax - ymin) * 0.04, lbl,
                rotation=90, fontsize=7.5, color=c,
                va="top", ha="right", alpha=0.85)


# --------------------------------------------------------------------------
# Figure 4: CPCV distribution (DCA)
# --------------------------------------------------------------------------

def fig4_cpcv_distribution(pickle_path: Path) -> Optional[plt.Figure]:
    """Render rolling-window XIRR distributions if a CPCV-DCA pickle exists.

    Expected pickle schema (best-effort; tolerant of two common shapes):
        {
          "windows": {5: <DataFrame>, 10: <DataFrame>, 15: <DataFrame>},
          ...
        }
      where each DataFrame has columns
        ['strategy', 'window_years', 'xirr']  (long form)
      OR
        {strategy_name: pd.Series of XIRR values}  (wide form)

    If the pickle is missing or malformed, returns None and writes a placeholder
    PNG explaining the situation (so downstream consumers don't 404).
    """
    out = FIG_DIR / "fig4_cpcv_dca_distribution.png"

    if not pickle_path.exists():
        return _fig4_placeholder(
            out, f"results/dca/{pickle_path.name} not found.\n"
                  "Run cpcv_dca.py to generate this distribution figure.")

    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return _fig4_placeholder(
            out, f"Could not read {pickle_path.name}: {e}")

    # Try to coerce into a long-form DataFrame
    long_df = _coerce_cpcv_to_long(data)
    if long_df is None or long_df.empty:
        return _fig4_placeholder(
            out, f"{pickle_path.name} did not contain XIRR samples in any\n"
                  "of the recognised shapes (windows->DataFrame, or\n"
                  "{strategy: Series}, or per-window dict-of-Series).")

    # Choose a comparison: Retail Holy Grail vs DCA QQQ B&H if present, else
    # the two strategies with the most samples.
    strategies = list(long_df["strategy"].unique())
    preferred = ("Retail Holy Grail", "DCA QQQ B&H")
    if all(p in strategies for p in preferred):
        s1, s2 = preferred
    else:
        # Fall back to the two with most samples
        counts = long_df.groupby("strategy")["xirr"].count().sort_values(ascending=False)
        s1 = counts.index[0]
        s2 = counts.index[1] if len(counts) > 1 else None

    windows = sorted(long_df["window_years"].unique())
    windows = [w for w in windows if w in (5, 10, 15)] or windows[:3]
    if not windows:
        return _fig4_placeholder(out, "No 5/10/15-yr windows in pickle.")

    fig, axes = plt.subplots(1, len(windows), figsize=(5 * len(windows), 5.5),
                              sharey=True)
    if len(windows) == 1:
        axes = [axes]

    for ax, w in zip(axes, windows):
        sub = long_df[long_df["window_years"] == w]
        groups, labels, colors = [], [], []
        for s, c in [(s1, "#b22222"), (s2, "#1f4e79")]:
            if s is None:
                continue
            vals = sub.loc[sub["strategy"] == s, "xirr"].dropna().values
            if len(vals) == 0:
                continue
            groups.append(vals * 100)
            labels.append(f"{s}\n(n={len(vals)}, "
                          f"med={np.median(vals)*100:.1f}%)")
            colors.append(c)
        if not groups:
            ax.text(0.5, 0.5, f"No data for {w}-yr window",
                     ha="center", va="center", transform=ax.transAxes)
            continue
        bp = ax.boxplot(groups, vert=True, patch_artist=True,
                         showmeans=True, widths=0.5,
                         meanprops=dict(marker="D", markerfacecolor="white",
                                          markeredgecolor="black", markersize=5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{w}-year rolling windows", fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        if ax is axes[0]:
            ax.set_ylabel("Annualised XIRR (%)", fontsize=10)

    fig.suptitle("Figure 4 — DCA-CPCV: Rolling-window XIRR Distributions",
                  fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[figures_dca] Saved {out}")
    return fig


def _coerce_cpcv_to_long(data) -> Optional[pd.DataFrame]:
    """Best-effort schema flattener for cpcv_dca pickle.

    Recognised shapes (in priority order):
      A) DataFrame with columns ``strategy/window_years/xirr``
      B) Dict of {"rolling_Nyr": DataFrame} where the DataFrame has
         columns ``retail_xirr`` and ``qqq_xirr`` (the cpcv_dca.py format)
      C) Dict {"results": ...} wrapper around any of the other shapes
      D) Dict keyed by window-year (int) -> DataFrame or {strategy: Series}
      E) Dict keyed by strategy -> {window_years: array-or-Series}
    """
    # Shape A
    if isinstance(data, pd.DataFrame):
        cols = set(data.columns)
        if {"strategy", "window_years", "xirr"} <= cols:
            return data[["strategy", "window_years", "xirr"]].copy()

    if not isinstance(data, dict):
        return None

    # Shape B: {"rolling_5yr": df, "rolling_10yr": df, "rolling_15yr": df, ...}
    rolling_keys = [k for k in data.keys()
                    if isinstance(k, str) and k.startswith("rolling_")
                    and k.endswith("yr")]
    if rolling_keys:
        rows = []
        # strategy column name (in pickle) -> output strategy label
        strategy_map = {
            "retail_xirr": "Retail Holy Grail",
            "qqq_xirr":    "DCA QQQ B&H",
            "tqqq_xirr":   "DCA TQQQ B&H",
            "spy_xirr":    "DCA SPY B&H",
            "lump_xirr":   "Lump-sum Rotation",
        }
        for k in rolling_keys:
            df = data[k]
            if not isinstance(df, pd.DataFrame):
                continue
            # parse window length from key like "rolling_5yr" -> 5
            try:
                w = int(k.replace("rolling_", "").replace("yr", ""))
            except ValueError:
                continue
            for col, label in strategy_map.items():
                if col in df.columns:
                    for v in df[col].dropna().values:
                        try:
                            rows.append({"strategy": label,
                                          "window_years": w,
                                          "xirr": float(v)})
                        except Exception:
                            continue
        if rows:
            return pd.DataFrame(rows)

    # Shape C: {"results": ...} wrapper -> recurse
    if "results" in data and isinstance(data["results"], (dict, pd.DataFrame)):
        rec = _coerce_cpcv_to_long(data["results"])
        if rec is not None and not rec.empty:
            return rec
    inner = data

    # Shape D: {window_years: DataFrame} or {window_years: {strategy: Series}}
    if all(isinstance(k, (int, float)) for k in inner.keys()):
        rows = []
        for w, payload in inner.items():
            if isinstance(payload, pd.DataFrame):
                cols = set(payload.columns)
                if {"strategy", "xirr"} <= cols:
                    for _, r in payload.iterrows():
                        rows.append({"strategy": r["strategy"],
                                      "window_years": int(w),
                                      "xirr": r["xirr"]})
                else:
                    for col in payload.columns:
                        for v in payload[col].dropna().values:
                            try:
                                rows.append({"strategy": str(col),
                                              "window_years": int(w),
                                              "xirr": float(v)})
                            except Exception:
                                continue
            elif isinstance(payload, dict):
                for s, ser in payload.items():
                    if isinstance(ser, pd.Series):
                        arr = ser.values
                    elif isinstance(ser, (list, tuple, np.ndarray)):
                        arr = np.asarray(list(ser))
                    else:
                        continue
                    for v in arr:
                        try:
                            rows.append({"strategy": str(s),
                                          "window_years": int(w),
                                          "xirr": float(v)})
                        except Exception:
                            continue
        if rows:
            return pd.DataFrame(rows)

    # Shape E: {strategy: {window_years: array-or-Series}} or {strategy: DataFrame}
    rows = []
    for s, payload in inner.items():
        if isinstance(payload, dict):
            for w, arr in payload.items():
                if isinstance(arr, pd.Series):
                    arr = arr.values
                elif isinstance(arr, (list, tuple, np.ndarray)):
                    arr = np.asarray(list(arr))
                else:
                    continue
                for v in arr:
                    try:
                        rows.append({"strategy": str(s),
                                      "window_years": int(w),
                                      "xirr": float(v)})
                    except Exception:
                        continue
        elif isinstance(payload, pd.DataFrame):
            cols = set(payload.columns)
            if {"window_years", "xirr"} <= cols:
                for _, r in payload.iterrows():
                    rows.append({"strategy": str(s),
                                  "window_years": int(r["window_years"]),
                                  "xirr": float(r["xirr"])})
    if rows:
        return pd.DataFrame(rows)
    return None


def _fig4_placeholder(out: Path, message: str) -> Optional[plt.Figure]:
    """Render a minimal placeholder PNG so downstream pipelines don't fail."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0.5, 0.7,
            "Figure 4 — DCA-CPCV: Rolling-window XIRR Distributions",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.45, message, ha="center", va="center", fontsize=10,
             color="#444", wrap=True)
    ax.text(0.5, 0.18,
            "(Placeholder — not a fatal error.)",
            ha="center", va="center", fontsize=9, color="#888", style="italic")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[figures_dca] Saved placeholder {out}")
    print(f"[figures_dca] Note: {message.splitlines()[0]}")
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    _ensure_dirs()
    print("[figures_dca] Loading universe ...")
    U = load_universe()
    print(f"  QQQ {U['qqq'].index[0].date()} -> {U['qqq'].index[-1].date()}"
          f" ({len(U['qqq'])} days)")

    # Suppress matplotlib's tight_layout warnings for log-scale + annotations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fig1 = fig1_equity_curves(U)
        plt.close(fig1)

        fig2 = fig2_dual_drawdown(U)
        plt.close(fig2)

        fig3 = fig3_r_cross_section(U)
        plt.close(fig3)

        fig4 = fig4_cpcv_distribution(CPCV_PKL)
        if fig4 is not None:
            plt.close(fig4)

    # Verify outputs
    expected = [
        FIG_DIR / "fig1_equity_curves_dca.png",
        FIG_DIR / "fig2_dual_drawdown.png",
        FIG_DIR / "fig3_r_cross_section.png",
        FIG_DIR / "fig4_cpcv_dca_distribution.png",
    ]
    print()
    print("[figures_dca] Outputs:")
    for p in expected:
        if p.exists():
            print(f"  OK  {p}  ({p.stat().st_size/1024:.0f} KB)")
        else:
            print(f"  MISSING  {p}")


if __name__ == "__main__":
    main()
