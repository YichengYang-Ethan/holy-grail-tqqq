"""
Peer Review (DCA-edition).

Re-runs the 7 robustness checks from `scripts/peer_review_fixes.py` but in the
DCA framework (retail Holy Grail). Lump-sum baselines are reproduced for parity
verification — every check that the original asked about lump-sum, we add the
DCA counterpart.

Tasks:
  1. Gayed 2016 baseline (canonical / variant / Kane-style / our blog) under
     identical retail cashflow ($10K + $10K/yr by default), plus DCA QQQ B&H
     and DCA TQQQ B&H benchmarks; bear-leg-premium isolation.
  2. Parameter robustness grid {SMA,EMA} × {5,10,20,50} × {150,200,250} under
     retail cashflow, with median/IQR + EMA(5,200) rank.
  3. Deflated Sharpe (Bailey-Lopez de Prado 2014) on TWR-derived Sharpe, scanned
     across N ∈ {1, 10, 24, 100, 500, 1000} trials.
  4. Start-date sensitivity (1999/2003/2010): retail / Gayed / DCA QQQ B&H.
  5. Volatility-drag formula (Avellaneda-Zhang) + DCA-specific empirical study:
     does DCA cashflow attenuate drag damage on a constant-long TQQQ position?
  6. Tax / slippage stress test: (fee_bps, slip_bps) ∈ {(0,0),(2.5,5),(5,10),
     (10,20),(25,50)} on retail strategy.
  7. DCA-specific: starting-regime sensitivity (peak / trough / mid) — 10-year
     retail backtests anchored at 2000-03-10, 2009-03-09, 2015-01-02.

Usage:
    python3 -m scripts.dca.peer_review_dca

Pickles results to `results/dca/peer_review_dca_results.pkl`.
"""
from __future__ import annotations

import os
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .core import (
    BacktestResult,
    CashflowPlan,
    backtest_dca,
    load_universe,
    make_lump_cashflow,
    make_retail_cashflow,
    metrics_dca,
    signal_ema,
    signal_price_vs_sma,
    signal_sma,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _print_header(num: int, title: str) -> None:
    print("\n" + "=" * 100)
    print(f"【PEER REVIEW DCA #{num}】{title}")
    print("=" * 100)


def _print_subheader(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def _bh_metrics(price: pd.Series, cashflow: CashflowPlan,
                fee_bps: float = 2.5, slip_bps: float = 5.0,
                asset_name: str = "asset") -> dict:
    """DCA buy-and-hold = always-on signal, single-asset universe.
    We piggyback on backtest_dca by using the same price for on/off and a
    signal of all-1 (so the strategy is always in `price`). Rotation count
    will be 0 since signal never flips."""
    sig = pd.Series(1.0, index=price.index)
    res = backtest_dca(price, price, sig, cashflow,
                       fee_bps=fee_bps, slip_bps=slip_bps,
                       asset_on_name=asset_name, asset_off_name=asset_name)
    m = metrics_dca(res)
    m["label"] = f"{asset_name} DCA B&H"
    return m


def _lump_compounding_metrics(price_on: pd.Series, price_off: pd.Series,
                               signal: pd.Series, init: float = 10_000.0,
                               fee_bps: float = 2.5, slip_bps: float = 5.0) -> dict:
    """Reproduces peer_review_fixes.backtest + metrics: pure return-compounding,
    one upfront $init, no cashflows. Used to verify parity with the original
    peer review numbers (so we know lump-sum results match)."""
    pos_l = signal.shift(1).fillna(0)
    off_pos_l = (1 - pos_l)
    on_ret = price_on.pct_change().fillna(0)
    off_ret = price_off.pct_change().fillna(0)
    pos_change = pos_l.diff().abs().fillna(0) * 2
    cost = pos_change * (fee_bps + slip_bps) / 10000.0
    strat_ret = pos_l * on_ret + off_pos_l * off_ret - cost
    eq = (1 + strat_ret).cumprod() * init
    if eq.empty or eq.iloc[-1] <= 0:
        return {"cagr": float("nan"), "mdd": float("nan"),
                "sharpe": float("nan"), "final": float("nan")}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    ret = eq.pct_change().fillna(0)
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    return {"cagr": float(cagr), "mdd": mdd, "sharpe": float(sh),
            "final": float(eq.iloc[-1])}


def _row_str(m: dict, name: str = None, w: int = 50) -> str:
    n = name if name is not None else m.get("label", "")
    return (f"{n:<{w}}"
            f" XIRR {m['xirr']*100:>6.2f}%"
            f" TWR {m['twr']*100:>6.2f}%"
            f" MDD-mkt {m['mdd_market']*100:>7.2f}%"
            f" MDD-contrib {m['mdd_vs_contrib']*100:>7.2f}%"
            f" Sharpe {m['sharpe']:>5.3f}"
            f" Final ${m['final_value']:>13,.0f}"
            f" Mult {m['multiple']:>5.2f}x")


# ===========================================================================
# REVIEW 1: Gayed 2016 baseline + bear-leg premium isolation
# ===========================================================================

def review_1_gayed(U: dict, initial: float, annual: float) -> dict:
    _print_header(1, "Gayed 2016 baseline (DCA edition)")

    qqq = U["qqq"]
    tqqq = U["tqqq"]
    tbill = U["tbill"]

    cf_retail = make_retail_cashflow(initial, annual, qqq.index, contrib_freq="M")
    print(f"\n  Retail cashflow plan: {cf_retail.label}")
    print(f"  Total to be contributed: ${cf_retail.total():,.0f}")

    sig_gayed = signal_price_vs_sma(qqq, 200)
    sig_ema = signal_ema(qqq, 5, 200)

    # Four core variants — DCA
    variants = []
    for label, sig, off in [
        ("Gayed (P>SMA200) → T-bills [CANONICAL]", sig_gayed, tbill),
        ("Gayed (P>SMA200) → QQQ      [VARIANT ]", sig_gayed, qqq),
        ("EMA(5,200)     → T-bills    [Kane    ]", sig_ema, tbill),
        ("EMA(5,200)     → QQQ        [BLOG    ]", sig_ema, qqq),
    ]:
        res = backtest_dca(tqqq, off, sig, cf_retail,
                            asset_on_name="TQQQ",
                            asset_off_name="QQQ" if off is qqq else "Tbill")
        m = metrics_dca(res)
        m["label_short"] = label
        variants.append(m)

    # DCA buy-and-hold benchmarks
    cf_qqq_bh = make_retail_cashflow(initial, annual, qqq.index, contrib_freq="M")
    bh_qqq = _bh_metrics(qqq, cf_qqq_bh, asset_name="QQQ")
    bh_tqqq = _bh_metrics(tqqq, cf_qqq_bh, asset_name="TQQQ")

    print()
    print(f"  {'Strategy':<46} {'XIRR':>8} {'TWR':>8} {'MDD-mkt':>9} {'MDD-c':>8}"
          f" {'Sharpe':>7} {'Final':>13} {'Mult':>6}")
    print("-" * 110)
    for m in variants:
        print(f"  {m['label_short']:<46}"
              f" {m['xirr']*100:>7.2f}% {m['twr']*100:>7.2f}%"
              f" {m['mdd_market']*100:>8.2f}% {m['mdd_vs_contrib']*100:>7.2f}%"
              f" {m['sharpe']:>7.3f} ${m['final_value']:>12,.0f} {m['multiple']:>5.2f}x")
    print(f"  {'QQQ DCA B&H':<46}"
          f" {bh_qqq['xirr']*100:>7.2f}% {bh_qqq['twr']*100:>7.2f}%"
          f" {bh_qqq['mdd_market']*100:>8.2f}% {bh_qqq['mdd_vs_contrib']*100:>7.2f}%"
          f" {bh_qqq['sharpe']:>7.3f} ${bh_qqq['final_value']:>12,.0f}"
          f" {bh_qqq['multiple']:>5.2f}x")
    print(f"  {'TQQQ DCA B&H (synth+real)':<46}"
          f" {bh_tqqq['xirr']*100:>7.2f}% {bh_tqqq['twr']*100:>7.2f}%"
          f" {bh_tqqq['mdd_market']*100:>8.2f}% {bh_tqqq['mdd_vs_contrib']*100:>7.2f}%"
          f" {bh_tqqq['sharpe']:>7.3f} ${bh_tqqq['final_value']:>12,.0f}"
          f" {bh_tqqq['multiple']:>5.2f}x")

    # Bear-leg premium isolation (DCA)
    g_tbill, g_qqq = variants[0], variants[1]
    e_tbill, e_qqq = variants[2], variants[3]
    print(f"\n  DCA bear-leg premium (1xQQQ vs T-bills, EMA 5/200):")
    print(f"    XIRR    {(e_qqq['xirr'] - e_tbill['xirr']) * 100:+.2f}pp")
    print(f"    TWR     {(e_qqq['twr'] - e_tbill['twr']) * 100:+.2f}pp")
    print(f"    MDD-mkt {(e_qqq['mdd_market'] - e_tbill['mdd_market']) * 100:+.2f}pp"
          f" (negative=worse)")
    print(f"    Sharpe  {e_qqq['sharpe'] - e_tbill['sharpe']:+.3f}")
    print(f"  DCA bear-leg premium (1xQQQ vs T-bills, P>SMA200):")
    print(f"    XIRR    {(g_qqq['xirr'] - g_tbill['xirr']) * 100:+.2f}pp")
    print(f"    TWR     {(g_qqq['twr'] - g_tbill['twr']) * 100:+.2f}pp")
    print(f"    MDD-mkt {(g_qqq['mdd_market'] - g_tbill['mdd_market']) * 100:+.2f}pp")
    print(f"    Sharpe  {g_qqq['sharpe'] - g_tbill['sharpe']:+.3f}")

    # Lump-sum parity check (must match original peer_review_fixes.py)
    _print_subheader("Lump-sum parity (must match original peer_review_fixes.py)")
    lump_qqq = _lump_compounding_metrics(tqqq, qqq, sig_ema)
    lump_tb = _lump_compounding_metrics(tqqq, tbill, sig_ema)
    g_lump_tb = _lump_compounding_metrics(tqqq, tbill, sig_gayed)
    g_lump_qqq = _lump_compounding_metrics(tqqq, qqq, sig_gayed)
    print(f"  EMA(5,200) → QQQ      lump CAGR={lump_qqq['cagr']*100:.2f}%"
          f" MDD={lump_qqq['mdd']*100:.2f}% Final=${lump_qqq['final']:,.0f}"
          f" (orig 15.88%/-95.50%/543,386)")
    print(f"  EMA(5,200) → T-bills  lump CAGR={lump_tb['cagr']*100:.2f}%"
          f" MDD={lump_tb['mdd']*100:.2f}% Final=${lump_tb['final']:,.0f}"
          f" (orig 15.78%/-83.58%/530,157)")
    print(f"  Gayed → T-bills       lump CAGR={g_lump_tb['cagr']*100:.2f}%"
          f" MDD={g_lump_tb['mdd']*100:.2f}% Final=${g_lump_tb['final']:,.0f}"
          f" (orig 10.02%/-95.16%/133,081)")
    print(f"  Gayed → QQQ           lump CAGR={g_lump_qqq['cagr']*100:.2f}%"
          f" MDD={g_lump_qqq['mdd']*100:.2f}% Final=${g_lump_qqq['final']:,.0f}"
          f" (orig 12.01%/-98.11%/216,274)")

    return {
        "cashflow_label": cf_retail.label,
        "total_contributed": cf_retail.total(),
        "variants": variants,
        "bh_qqq": bh_qqq,
        "bh_tqqq": bh_tqqq,
        "lump_parity": {
            "ema_qqq": lump_qqq, "ema_tbill": lump_tb,
            "gayed_tbill": g_lump_tb, "gayed_qqq": g_lump_qqq,
        },
        "bear_leg_premium_dca_ema": {
            "xirr_pp": (e_qqq['xirr'] - e_tbill['xirr']),
            "twr_pp": (e_qqq['twr'] - e_tbill['twr']),
            "mdd_mkt_pp": (e_qqq['mdd_market'] - e_tbill['mdd_market']),
            "sharpe": (e_qqq['sharpe'] - e_tbill['sharpe']),
        },
        "bear_leg_premium_dca_gayed": {
            "xirr_pp": (g_qqq['xirr'] - g_tbill['xirr']),
            "twr_pp": (g_qqq['twr'] - g_tbill['twr']),
            "mdd_mkt_pp": (g_qqq['mdd_market'] - g_tbill['mdd_market']),
            "sharpe": (g_qqq['sharpe'] - g_tbill['sharpe']),
        },
    }


# ===========================================================================
# REVIEW 2: Parameter robustness grid
# ===========================================================================

def review_2_param_grid(U: dict, initial: float, annual: float) -> dict:
    _print_header(2, "Param robustness grid {SMA,EMA}×{5,10,20,50}×{150,200,250}")

    qqq = U["qqq"]
    tqqq = U["tqqq"]
    cf = make_retail_cashflow(initial, annual, qqq.index, contrib_freq="M")
    print(f"\n  Cashflow: {cf.label}")
    print(f"  Bear leg = QQQ; total contributed ${cf.total():,.0f}")

    rows = []
    print(f"\n  {'MA':<5}{'F':>4}{'S':>5}"
          f"{'XIRR':>9}{'TWR':>9}{'MDD-mkt':>10}{'Sharpe':>9}{'Final':>15}")
    print("  " + "-" * 75)
    for ma_type in ["EMA", "SMA"]:
        for fast in [5, 10, 20, 50]:
            for slow in [150, 200, 250]:
                if fast >= slow:
                    continue
                sig = signal_ema(qqq, fast, slow) if ma_type == "EMA" else signal_sma(qqq, fast, slow)
                res = backtest_dca(tqqq, qqq, sig, cf,
                                    asset_on_name="TQQQ", asset_off_name="QQQ")
                m = metrics_dca(res)
                star = " *" if (ma_type, fast, slow) == ("EMA", 5, 200) else ""
                rows.append({
                    "ma_type": ma_type, "fast": fast, "slow": slow,
                    "xirr": m["xirr"], "twr": m["twr"],
                    "mdd_market": m["mdd_market"], "mdd_vs_contrib": m["mdd_vs_contrib"],
                    "sharpe": m["sharpe"], "calmar_xirr": m["calmar_xirr"],
                    "final_value": m["final_value"], "multiple": m["multiple"],
                    "n_rotations": m["n_rotations"],
                })
                print(f"  {ma_type:<5}{fast:>4}{slow:>5}"
                      f"{m['xirr']*100:>8.2f}%"
                      f"{m['twr']*100:>8.2f}%"
                      f"{m['mdd_market']*100:>9.2f}%"
                      f"{m['sharpe']:>9.3f}"
                      f"${m['final_value']:>13,.0f}{star}")

    df = pd.DataFrame(rows)

    print("\n  Grid summary (N=24):")
    print(f"    XIRR     median {df['xirr'].median()*100:>6.2f}%,"
          f" IQR [{df['xirr'].quantile(0.25)*100:.2f}%, {df['xirr'].quantile(0.75)*100:.2f}%],"
          f" min {df['xirr'].min()*100:.2f}%, max {df['xirr'].max()*100:.2f}%")
    print(f"    TWR      median {df['twr'].median()*100:>6.2f}%,"
          f" IQR [{df['twr'].quantile(0.25)*100:.2f}%, {df['twr'].quantile(0.75)*100:.2f}%]")
    print(f"    MDD-mkt  median {df['mdd_market'].median()*100:>6.2f}%,"
          f" IQR [{df['mdd_market'].quantile(0.25)*100:.2f}%, {df['mdd_market'].quantile(0.75)*100:.2f}%]")
    print(f"    Sharpe   median {df['sharpe'].median():>6.3f},"
          f" IQR [{df['sharpe'].quantile(0.25):.3f}, {df['sharpe'].quantile(0.75):.3f}]")

    # Where does EMA(5,200) sit?
    target = df[(df.ma_type == "EMA") & (df.fast == 5) & (df.slow == 200)].iloc[0]
    rank_xirr = int((df["xirr"] > target["xirr"]).sum() + 1)
    rank_twr = int((df["twr"] > target["twr"]).sum() + 1)
    rank_sharpe = int((df["sharpe"] > target["sharpe"]).sum() + 1)
    print(f"\n  EMA(5,200) rank under DCA:"
          f"  XIRR #{rank_xirr}/{len(df)}"
          f"  TWR #{rank_twr}/{len(df)}"
          f"  Sharpe #{rank_sharpe}/{len(df)}")
    if rank_xirr == 1:
        print("    >> #1 → suggests overfitting to in-sample.")
    elif rank_xirr >= len(df) - 4:
        print("    >> bottom-quintile → parameter NOT robust under DCA.")
    else:
        print("    >> mid-tier → parameter is plausibly robust under DCA.")

    return {
        "grid": rows,
        "df": df,
        "ema_5_200_rank": {
            "xirr": rank_xirr, "twr": rank_twr, "sharpe": rank_sharpe,
            "n_total": len(df),
        },
        "summary": {
            "xirr_median": float(df["xirr"].median()),
            "xirr_iqr": (float(df["xirr"].quantile(0.25)),
                         float(df["xirr"].quantile(0.75))),
            "xirr_min": float(df["xirr"].min()),
            "xirr_max": float(df["xirr"].max()),
            "twr_median": float(df["twr"].median()),
            "mdd_market_median": float(df["mdd_market"].median()),
            "sharpe_median": float(df["sharpe"].median()),
        },
    }


# ===========================================================================
# REVIEW 3: Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014)
# ===========================================================================

def review_3_deflated_sharpe(U: dict, initial: float, annual: float) -> dict:
    _print_header(3, "Deflated Sharpe Ratio (DCA TWR-derived)")

    qqq = U["qqq"]; tqqq = U["tqqq"]
    sig = signal_ema(qqq, 5, 200)
    cf = make_retail_cashflow(initial, annual, qqq.index, contrib_freq="M")
    res = backtest_dca(tqqq, qqq, sig, cf,
                        asset_on_name="TQQQ", asset_off_name="QQQ")
    m = metrics_dca(res)
    sr = m["sharpe"]
    years = (qqq.index[-1] - qqq.index[0]).days / 365.25
    print(f"\n  DCA-strategy Sharpe (TWR sub-period): {sr:.3f}")
    print(f"  Years: {years:.1f}")

    def deflated_sharpe(sr: float, N: int, T_years: float,
                        sr_std: float = 0.5, skew: float = 0,
                        kurt: float = 3) -> tuple:
        """Mirror peer_review_fixes.deflated_sharpe exactly."""
        gamma = 0.5772
        if N <= 1:
            sr0 = float("-inf")
        else:
            sr0 = sr_std * ((1 - gamma) * norm.ppf(1 - 1 / N) +
                            gamma * norm.ppf(1 - 1 / (N * np.e)))
        T = T_years * 252
        denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
        if denom <= 0:
            return 0.0, sr0
        if N <= 1:
            dsr = 1.0
        else:
            dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
        return float(dsr), float(sr0)

    print(f"\n  {'N':<6}{'SR* null':>12}{'DSR':>14}{'verdict':>11}")
    print("  " + "-" * 50)
    rows = []
    for N in [1, 10, 24, 100, 500, 1000]:
        dsr, sr0 = deflated_sharpe(sr, N, years)
        verdict = "STRONG" if dsr > 0.95 else ("MARGIN" if dsr > 0.75 else "WEAK")
        rows.append({"N": N, "sr_null": sr0, "dsr": dsr, "verdict": verdict})
        sr0_str = "-inf" if not np.isfinite(sr0) else f"{sr0:.3f}"
        print(f"  N={N:<5}{sr0_str:>12}{dsr*100:>13.1f}%{verdict:>11}")

    print("\n  Interpretation:")
    print("    DSR ≥ 0.95: strong evidence true SR > 0")
    print("    DSR 0.75–0.95: marginal")
    print("    DSR < 0.75: indistinguishable from luck given assumed search size")

    return {"sharpe_dca": sr, "years": years, "rows": rows}


# ===========================================================================
# REVIEW 4: Start-date sensitivity
# ===========================================================================

def review_4_start_date(U: dict, initial: float = 10_000.0,
                          annual: float = 10_000.0) -> dict:
    _print_header(4, "Start-date sensitivity (DCA): 1999 / 2003 / 2010")

    qqq = U["qqq"]; tqqq = U["tqqq"]; tbill = U["tbill"]
    starts = ["1999-03-10", "2003-01-02", "2010-02-11"]

    print(f"\n  {'Start':<13}{'Strategy':<30}"
          f"{'XIRR':>8}{'TWR':>8}{'MDD':>9}{'Total in':>12}{'Final':>14}{'Mult':>7}")
    print("  " + "-" * 105)
    rows = []
    for sd in starts:
        idx = qqq.index[qqq.index >= pd.Timestamp(sd)]

        # Retail (EMA 5/200 → QQQ)
        cf = make_retail_cashflow(initial, annual, idx, contrib_freq="M")
        sig_e = signal_ema(qqq.loc[idx], 5, 200)
        res = backtest_dca(tqqq.loc[idx], qqq.loc[idx], sig_e, cf,
                            asset_on_name="TQQQ", asset_off_name="QQQ")
        m_retail = metrics_dca(res)
        m_retail["start"] = sd; m_retail["strategy"] = "Retail (EMA 5/200 → QQQ)"
        print(f"  {sd:<13}{'Retail (EMA 5/200 → QQQ)':<30}"
              f"{m_retail['xirr']*100:>7.2f}%"
              f"{m_retail['twr']*100:>7.2f}%"
              f"{m_retail['mdd_market']*100:>8.2f}%"
              f"${cf.total():>11,.0f}"
              f"${m_retail['final_value']:>13,.0f}"
              f"{m_retail['multiple']:>6.2f}x")

        # Gayed (P>SMA200 → T-bills)
        cf_g = make_retail_cashflow(initial, annual, idx, contrib_freq="M")
        sig_g = signal_price_vs_sma(qqq.loc[idx], 200)
        res_g = backtest_dca(tqqq.loc[idx], tbill.loc[idx], sig_g, cf_g,
                              asset_on_name="TQQQ", asset_off_name="Tbill")
        m_g = metrics_dca(res_g); m_g["start"] = sd
        m_g["strategy"] = "Gayed (P>SMA200 → T-bills)"
        print(f"  {sd:<13}{'Gayed (P>SMA200 → T-bills)':<30}"
              f"{m_g['xirr']*100:>7.2f}%"
              f"{m_g['twr']*100:>7.2f}%"
              f"{m_g['mdd_market']*100:>8.2f}%"
              f"${cf_g.total():>11,.0f}"
              f"${m_g['final_value']:>13,.0f}"
              f"{m_g['multiple']:>6.2f}x")

        # DCA QQQ B&H
        cf_bh = make_retail_cashflow(initial, annual, idx, contrib_freq="M")
        m_bh = _bh_metrics(qqq.loc[idx], cf_bh, asset_name="QQQ")
        m_bh["start"] = sd; m_bh["strategy"] = "QQQ DCA B&H"
        print(f"  {sd:<13}{'QQQ DCA B&H':<30}"
              f"{m_bh['xirr']*100:>7.2f}%"
              f"{m_bh['twr']*100:>7.2f}%"
              f"{m_bh['mdd_market']*100:>8.2f}%"
              f"${cf_bh.total():>11,.0f}"
              f"${m_bh['final_value']:>13,.0f}"
              f"{m_bh['multiple']:>6.2f}x")
        print()
        rows.extend([m_retail, m_g, m_bh])

    return {"rows": rows}


# ===========================================================================
# REVIEW 5: Volatility drag formula + DCA-specific empirical
# ===========================================================================

def review_5_volatility_drag(U: dict) -> dict:
    _print_header(5, "Volatility drag (Avellaneda-Zhang) + DCA empirical")

    qqq = U["qqq"]; tqqq = U["tqqq"]; tqqq_real = U["tqqq_real"]

    L = 3
    print(f"\n  Formula: drag ≈ (L²-L)/2 × σ² = {(L**2-L)/2:.1f} × σ² (L=3)")
    print(f"\n  {'σ':>7}{'drag (%/yr)':>15}")
    drag_table = []
    for sigma in [0.15, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50]:
        drag = (L ** 2 - L) / 2 * sigma ** 2
        drag_table.append({"sigma": sigma, "drag_yr": drag})
        print(f"  {sigma*100:>6.0f}%{drag*100:>13.2f}%")

    # 2010-2026 empirical drag
    overlap_idx = tqqq_real.index
    qqq_real_vol = qqq.loc[overlap_idx[0]:overlap_idx[-1]].pct_change().std() * np.sqrt(252)
    expected_drag = 3 * qqq_real_vol ** 2
    print(f"\n  Empirical (2010-02-11 → today):")
    print(f"    QQQ realized vol: {qqq_real_vol*100:.1f}%")
    print(f"    Expected drag (L=3): {expected_drag*100:.2f}%/yr")

    # === DCA-specific: does the cashflow attenuate drag damage? ===
    # Constant-long TQQQ position, lump vs DCA. signal=all 1, no T-bill leg.
    # Compare lump-sum vs DCA-retail final-value-per-dollar-contributed.
    _print_subheader("DCA-vs-lump on constant-long TQQQ (signal=1 forever)")

    sig_long = pd.Series(1.0, index=qqq.index)

    # Run on full history (1999-) and on 2010+ (real TQQQ era only)
    for label, slice_idx in [
        ("Full history (1999-2026, synth+real)", qqq.index),
        ("Real TQQQ only (2010-2026)", overlap_idx),
    ]:
        # Lump-sum: $100K on day 0
        cf_lump = make_lump_cashflow(100_000.0, slice_idx)
        res_lump = backtest_dca(tqqq.reindex(slice_idx),
                                  tqqq.reindex(slice_idx),
                                  sig_long.reindex(slice_idx).fillna(1.0),
                                  cf_lump,
                                  asset_on_name="TQQQ", asset_off_name="TQQQ")
        m_lump = metrics_dca(res_lump)

        # DCA-retail: $10K + $10K/yr matched
        cf_dca = make_retail_cashflow(10_000, 10_000, slice_idx, contrib_freq="M")
        res_dca = backtest_dca(tqqq.reindex(slice_idx),
                                tqqq.reindex(slice_idx),
                                sig_long.reindex(slice_idx).fillna(1.0),
                                cf_dca,
                                asset_on_name="TQQQ", asset_off_name="TQQQ")
        m_dca = metrics_dca(res_dca)

        print(f"\n  {label}")
        print(f"    Lump   $100K → final ${m_lump['final_value']:,.0f}"
              f"  XIRR {m_lump['xirr']*100:.2f}%"
              f"  MDD-mkt {m_lump['mdd_market']*100:.2f}%"
              f"  per $1 contributed: ${m_lump['multiple']:.4f}")
        print(f"    DCA   $10K+10K/yr (${cf_dca.total():,.0f}) → final ${m_dca['final_value']:,.0f}"
              f"  XIRR {m_dca['xirr']*100:.2f}%"
              f"  MDD-mkt {m_dca['mdd_market']*100:.2f}%"
              f"  per $1 contributed: ${m_dca['multiple']:.4f}")
        print(f"    >> DCA Mult / Lump Mult = {m_dca['multiple']/m_lump['multiple']:.3f}"
              f" (>1.00 means DCA blunts drag damage)")
        print(f"    >> DCA MDD-mkt − Lump MDD-mkt = "
              f"{(m_dca['mdd_market']-m_lump['mdd_market'])*100:+.2f}pp"
              f" (positive = DCA had milder MV drawdown)")

    return {
        "drag_table": drag_table,
        "qqq_vol_2010_26": float(qqq_real_vol),
        "expected_drag_2010_26": float(expected_drag),
    }


# ===========================================================================
# REVIEW 6: Cost stress test
# ===========================================================================

def review_6_cost_stress(U: dict, initial: float, annual: float) -> dict:
    _print_header(6, "Cost stress test (DCA): fee_bps + slip_bps grid")

    qqq = U["qqq"]; tqqq = U["tqqq"]
    cf = make_retail_cashflow(initial, annual, qqq.index, contrib_freq="M")
    sig = signal_ema(qqq, 5, 200)

    grid = [(0, 0), (2.5, 5), (5, 10), (10, 20), (25, 50)]
    print(f"\n  {'fee':>6}{'slip':>6}"
          f"{'XIRR':>9}{'TWR':>9}{'MDD-mkt':>10}{'Sharpe':>9}{'Final':>14}{'Rot':>6}")
    print("  " + "-" * 75)
    rows = []
    for fee_bps, slip_bps in grid:
        res = backtest_dca(tqqq, qqq, sig, cf,
                            fee_bps=fee_bps, slip_bps=slip_bps,
                            asset_on_name="TQQQ", asset_off_name="QQQ")
        m = metrics_dca(res)
        rows.append({"fee_bps": fee_bps, "slip_bps": slip_bps,
                     "xirr": m["xirr"], "twr": m["twr"],
                     "mdd_market": m["mdd_market"],
                     "sharpe": m["sharpe"], "final_value": m["final_value"],
                     "n_rotations": m["n_rotations"]})
        print(f"  {fee_bps:>6.1f}{slip_bps:>6.1f}"
              f"{m['xirr']*100:>8.2f}%"
              f"{m['twr']*100:>8.2f}%"
              f"{m['mdd_market']*100:>9.2f}%"
              f"{m['sharpe']:>9.3f}"
              f"${m['final_value']:>13,.0f}"
              f"{m['n_rotations']:>6}")

    base = rows[1]  # baseline = (2.5, 5)
    worst = rows[-1]  # (25, 50)
    print(f"\n  Worst (fee=25bps,slip=50bps) vs base (fee=2.5bps,slip=5bps):")
    print(f"    ΔXIRR    {(worst['xirr']-base['xirr'])*100:+.2f}pp")
    print(f"    ΔTWR     {(worst['twr']-base['twr'])*100:+.2f}pp")
    print(f"    ΔSharpe  {worst['sharpe']-base['sharpe']:+.3f}")
    print(f"    ΔFinal   ${worst['final_value']-base['final_value']:+,.0f}")

    return {"rows": rows}


# ===========================================================================
# REVIEW 7: Starting-regime sensitivity (DCA-specific)
# ===========================================================================

def review_7_starting_regime(U: dict, initial: float = 10_000.0,
                                annual: float = 10_000.0,
                                years: int = 10) -> dict:
    _print_header(7, f"Starting-regime sensitivity ({years}-yr retail DCA)")

    qqq = U["qqq"]; tqqq = U["tqqq"]
    regimes = [
        ("2000-03-10", "dot-com peak"),
        ("2009-03-09", "GFC bottom"),
        ("2015-01-02", "mid-cycle"),
    ]

    print(f"\n  Each run: ${initial:,.0f} initial + ${annual:,.0f}/yr DCA, {years}-yr window")
    print()
    print(f"  {'Start':<13}{'Regime':<16}"
          f"{'XIRR':>8}{'TWR':>8}{'MDD-mkt':>10}{'MDD-c':>10}{'Total in':>13}{'Final':>14}{'Mult':>7}")
    print("  " + "-" * 100)

    rows = []
    for sd, regime in regimes:
        start_ts = pd.Timestamp(sd)
        end_ts = start_ts + pd.DateOffset(years=years)
        idx = qqq.index[(qqq.index >= start_ts) & (qqq.index <= end_ts)]
        if len(idx) < 200:
            print(f"  {sd:<13}{regime:<16}  (insufficient data)")
            continue
        cf = make_retail_cashflow(initial, annual, idx, contrib_freq="M")
        sig = signal_ema(qqq.loc[idx], 5, 200)
        res = backtest_dca(tqqq.loc[idx], qqq.loc[idx], sig, cf,
                            asset_on_name="TQQQ", asset_off_name="QQQ")
        m = metrics_dca(res); m["start"] = sd; m["regime"] = regime
        m["total_contributed_actual"] = cf.total()
        rows.append(m)
        print(f"  {sd:<13}{regime:<16}"
              f"{m['xirr']*100:>7.2f}%"
              f"{m['twr']*100:>7.2f}%"
              f"{m['mdd_market']*100:>9.2f}%"
              f"{m['mdd_vs_contrib']*100:>9.2f}%"
              f"${cf.total():>11,.0f}"
              f"${m['final_value']:>13,.0f}"
              f"{m['multiple']:>6.2f}x")

    # Spread analysis
    if len(rows) >= 2:
        xirrs = [r["xirr"] for r in rows]
        finals = [r["final_value"] for r in rows]
        mdds = [r["mdd_market"] for r in rows]
        print(f"\n  Cross-regime spread:")
        print(f"    XIRR    min {min(xirrs)*100:.2f}% / max {max(xirrs)*100:.2f}%"
              f" (range {(max(xirrs)-min(xirrs))*100:.2f}pp)")
        print(f"    Final   min ${min(finals):,.0f} / max ${max(finals):,.0f}"
              f" (ratio {max(finals)/min(finals):.2f}x)")
        print(f"    MDD-mkt min {min(mdds)*100:.2f}% / max {max(mdds)*100:.2f}%"
              f" (range {(max(mdds)-min(mdds))*100:.2f}pp)")

    return {"rows": rows}


# ===========================================================================
# Main
# ===========================================================================

def main() -> dict:
    print("[peer_review_dca] Loading universe ...")
    U = load_universe()
    print(f"  QQQ {U['qqq'].index[0].date()} → {U['qqq'].index[-1].date()}"
          f"  ({len(U['qqq'])} days)")

    INITIAL = 10_000.0
    ANNUAL = 10_000.0

    out = {}
    out["1_gayed"] = review_1_gayed(U, INITIAL, ANNUAL)
    out["2_param_grid"] = review_2_param_grid(U, INITIAL, ANNUAL)
    out["3_deflated_sharpe"] = review_3_deflated_sharpe(U, INITIAL, ANNUAL)
    out["4_start_date"] = review_4_start_date(U, INITIAL, ANNUAL)
    out["5_volatility_drag"] = review_5_volatility_drag(U)
    out["6_cost_stress"] = review_6_cost_stress(U, INITIAL, ANNUAL)
    out["7_starting_regime"] = review_7_starting_regime(U, INITIAL, ANNUAL, years=10)

    # Persist
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "results" / "dca"
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "peer_review_dca_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\n[OK] Pickled results -> {pkl_path}")

    # Final summary block
    print("\n" + "#" * 100)
    print("# DCA PEER REVIEW SUMMARY")
    print("#" * 100)
    g_blog = out["1_gayed"]["variants"][3]  # EMA(5,200) → QQQ
    g_canon = out["1_gayed"]["variants"][0]  # Gayed canon
    print(f"\n[1] Retail Holy Grail (EMA 5/200 → QQQ) DCA:"
          f" XIRR {g_blog['xirr']*100:.2f}%"
          f" / MDD-mkt {g_blog['mdd_market']*100:.2f}%"
          f" / MDD-c {g_blog['mdd_vs_contrib']*100:.2f}%"
          f" / Final ${g_blog['final_value']:,.0f} on ${out['1_gayed']['total_contributed']:,.0f} in.")
    print(f"    Gayed canonical baseline: XIRR {g_canon['xirr']*100:.2f}%, MDD-mkt"
          f" {g_canon['mdd_market']*100:.2f}%.")
    rk = out["2_param_grid"]["ema_5_200_rank"]
    print(f"\n[2] EMA(5,200) under DCA grid: XIRR rank #{rk['xirr']}/{rk['n_total']},"
          f" Sharpe rank #{rk['sharpe']}/{rk['n_total']}.")
    print(f"\n[3] DSR @ N=24 trials: see table above (DCA Sharpe = "
          f"{out['3_deflated_sharpe']['sharpe_dca']:.3f}).")
    print(f"\n[7] Regime sensitivity over 10-yr windows:")
    for r in out["7_starting_regime"]["rows"]:
        print(f"    {r['start']} ({r['regime']}): XIRR {r['xirr']*100:.2f}%,"
              f" MDD-mkt {r['mdd_market']*100:.2f}%, Final ${r['final_value']:,.0f}.")

    return out


if __name__ == "__main__":
    main()
