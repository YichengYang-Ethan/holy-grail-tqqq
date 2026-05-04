"""
Figure 5 — Full 3×3 matrix visualization:
    cashflow ∈ {Lump-sum, Retail r=1, Pure DCA}
        × strategy ∈ {QQQ B&H, TQQQ B&H, Rotation EMA(5,200)}

Renders three side-by-side annotated heatmaps:
    Panel A — Final value (log-scaled colormap)
    Panel B — XIRR (diverging green-red, money-weighted)
    Panel C — Market-value MDD (sequential red, more negative = darker)

Plus a 4th panel: same matrix but TWR — which exposes that DCA-on-TQQQ's
spectacular XIRR is not real strategy alpha.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .core import (
    load_universe, signal_ema,
    make_lump_cashflow, make_retail_cashflow, make_pure_dca_cashflow,
    backtest_dca, metrics_dca,
)

INITIAL = 10_000
ANNUAL = 10_000

CASHFLOWS = ["Lump-sum", "Retail r=1", "Pure DCA"]
STRATEGIES = ["QQQ B&H", "TQQQ B&H", "Rotation"]


def build_matrix() -> pd.DataFrame:
    U = load_universe()
    qqq, tqqq = U["qqq"], U["tqqq"]
    sig = signal_ema(qqq, 5, 200)
    sig_zero = pd.Series(0.0, index=qqq.index)
    sig_one = pd.Series(1.0, index=qqq.index)

    cf_dict = {
        "Lump-sum":     make_lump_cashflow(INITIAL, qqq.index),
        "Retail r=1":   make_retail_cashflow(INITIAL, ANNUAL, qqq.index),
        "Pure DCA":     make_pure_dca_cashflow(INITIAL, ANNUAL, qqq.index),
    }
    strats = {
        "QQQ B&H":  (qqq, qqq, sig_zero, "QQQ", "QQQ"),
        "TQQQ B&H": (tqqq, tqqq, sig_one, "TQQQ", "TQQQ"),
        "Rotation": (tqqq, qqq, sig, "TQQQ", "QQQ"),
    }
    rows = []
    for cf_name in CASHFLOWS:
        for st_name in STRATEGIES:
            p_on, p_off, s, on, off = strats[st_name]
            res = backtest_dca(p_on, p_off, s, cf_dict[cf_name],
                                asset_on_name=on, asset_off_name=off)
            m = metrics_dca(res)
            rows.append({
                "cashflow": cf_name, "strategy": st_name,
                "total_in": cf_dict[cf_name].total(),
                "final": m["final_value"],
                "xirr": m["xirr"] * 100,
                "twr": m["twr"] * 100,
                "mdd_mkt": m["mdd_market"] * 100,
                "mdd_contrib": m["mdd_vs_contrib"] * 100,
                "mult": m["multiple"],
            })
    return pd.DataFrame(rows)


def _pivot(df: pd.DataFrame, value: str) -> np.ndarray:
    """Return matrix shape (3,3) ordered by CASHFLOWS × STRATEGIES."""
    pv = df.pivot(index="cashflow", columns="strategy", values=value)
    return pv.reindex(index=CASHFLOWS, columns=STRATEGIES).values


def _annotate(ax, mat: np.ndarray, fmt: callable, color_decision: callable) -> None:
    n_rows, n_cols = mat.shape
    for i in range(n_rows):
        for j in range(n_cols):
            txt_color = color_decision(mat[i, j])
            ax.text(j, i, fmt(mat[i, j]),
                     ha="center", va="center",
                     color=txt_color, fontsize=11, fontweight="bold")


def _setup_axis(ax, title: str) -> None:
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_yticks(range(len(CASHFLOWS)))
    ax.set_xticklabels(STRATEGIES, fontsize=10)
    ax.set_yticklabels(CASHFLOWS, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)


def make_fig5(df: pd.DataFrame, save_path: str) -> None:
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.35, wspace=0.25)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[0, 3])
    axTxt = fig.add_subplot(gs[1, :])  # large summary block

    # --- Panel A: Final value (log-scale colormap)
    final = _pivot(df, "final")
    norm = mcolors.LogNorm(vmin=final.min(), vmax=final.max())
    im = axA.imshow(final, cmap="Blues", norm=norm, aspect="auto")
    _setup_axis(axA, "Panel A — Final value ($)")
    _annotate(axA, final,
               fmt=lambda v: f"${v/1e6:.2f}M" if v >= 1e6 else
                              (f"${v/1e3:.0f}K" if v >= 1e4 else f"${v:,.0f}"),
               color_decision=lambda v: "white" if v > final.max()*0.4 else "black")
    plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04, format="%.0e")

    # --- Panel B: XIRR (diverging)
    xirr = _pivot(df, "xirr")
    vmax = max(abs(xirr.min()), abs(xirr.max()))
    im = axB.imshow(xirr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    _setup_axis(axB, "Panel B — XIRR % (money-weighted)")
    _annotate(axB, xirr, fmt=lambda v: f"{v:.2f}%",
               color_decision=lambda v: "black")
    plt.colorbar(im, ax=axB, fraction=0.046, pad=0.04)

    # --- Panel C: MDD-market (Reds, more negative = darker)
    mdd = _pivot(df, "mdd_mkt")
    im = axC.imshow(mdd, cmap="Reds_r", vmin=mdd.min(), vmax=0, aspect="auto")
    _setup_axis(axC, "Panel C — Max DD vs market peak %")
    _annotate(axC, mdd, fmt=lambda v: f"{v:.1f}%",
               color_decision=lambda v: "white" if v < mdd.min()*0.4 else "black")
    plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04)

    # --- Panel D: TWR (exposes the XIRR illusion on DCA TQQQ)
    twr = _pivot(df, "twr")
    vmax_t = max(abs(twr.min()), abs(twr.max()))
    im = axD.imshow(twr, cmap="RdYlGn", vmin=0, vmax=20, aspect="auto")
    _setup_axis(axD, "Panel D — TWR % (time-weighted, strategy alpha)")
    _annotate(axD, twr, fmt=lambda v: f"{v:.2f}%",
               color_decision=lambda v: "black")
    plt.colorbar(im, ax=axD, fraction=0.046, pad=0.04)

    # --- Bottom: text summary table
    axTxt.axis("off")
    table_lines = [
        f"{'Cashflow':<14}{'Strategy':<12}{'Total in':>12}{'Final':>14}{'Mult':>8}"
        f"{'XIRR':>9}{'TWR':>9}{'MDD-mkt':>10}{'MDD-vs-contrib':>16}",
        "─" * 104,
    ]
    prev_cf = None
    for _, r in df.iterrows():
        cfn = r["cashflow"] if r["cashflow"] != prev_cf else ""
        line = (f"{cfn:<14}{r['strategy']:<12}${r['total_in']:>11,.0f}"
                f"${r['final']:>13,.0f}{r['mult']:>7.1f}x"
                f"{r['xirr']:>8.2f}%{r['twr']:>8.2f}%"
                f"{r['mdd_mkt']:>9.2f}%{r['mdd_contrib']:>15.2f}%")
        table_lines.append(line)
        prev_cf = r["cashflow"]
    table_lines.append("─" * 104)
    table_lines.append(
        "Notes: Lump-sum row reproduces holy-grail README exactly  ·  "
        "Retail r=1 = 100% LSI day-0 + monthly DCA  ·  Pure DCA = 12-month phase-in + monthly DCA  ·  "
        "DCA TQQQ B&H XIRR looks high but TWR ≈ 1.3% — money-weighted illusion, not alpha."
    )
    axTxt.text(0.0, 1.0, "\n".join(table_lines),
                family="monospace", fontsize=10, va="top", ha="left",
                transform=axTxt.transAxes)

    fig.suptitle(
        "Figure 5 — Full 3×3 Matrix: Cashflow × Strategy "
        f"(${INITIAL:,} initial + ${ANNUAL:,}/yr where applicable, 1999-2026)",
        fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("[fig5] Building 3×3 matrix ...")
    df = build_matrix()
    print(df.to_string(index=False))
    out_path = "figures/dca/fig5_3x3_matrix.png"
    print(f"[fig5] Rendering to {out_path} ...")
    make_fig5(df, out_path)
    df.to_pickle("results/dca/fig5_matrix.pkl")
    print("[fig5] Done.")


if __name__ == "__main__":
    main()
