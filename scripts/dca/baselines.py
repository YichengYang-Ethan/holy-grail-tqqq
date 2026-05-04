"""
DCA Holy Grail — Baselines.

Runs 7 baseline strategies under the *same* cashflow plan as the retail Holy
Grail rotation, plus the original lump-sum EMA(5,200) rotation, so that any
gap between Retail Holy Grail and these baselines can be cleanly attributed
to (a) the rotation rule itself, (b) the r-rule cashflow, or (c) leverage.

Baselines:
    1. DCA QQQ B&H                 — full-period buy-and-hold QQQ
    2. DCA SPY B&H                 — full-period buy-and-hold SPY
    3. DCA TQQQ B&H                — full-period buy-and-hold TQQQ (synth+real)
    4. DCA 60/40 (QQQ + AGG)       — static 60/40 with annual rebalance
                                     (AGG synthesised from t-bill 1999-2003)
    5. DCA 60/40 (SPY + AGG)       — same with SPY/AGG
    6. Pure DCA Rotation           — EMA(5,200) rotation but with 12-month
                                     phase-in (no r-rule)
    7. Lump-sum Rotation           — original Holy Grail backtest, $10k LSI

For comparison, also runs the "Retail Holy Grail" rotation under the same
cashflow plan as 1-6 (so any gap = the rule itself, not contribution mechanics).

Do NOT modify core.py. The 60/40 mix isn't supported by core.backtest_dca
(which is binary B-case rotation), so it's implemented as a stand-alone
function `backtest_static_mix` in this file.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (
    BacktestResult,
    CashflowPlan,
    backtest_dca,
    fmt_metric_row,
    load_universe,
    make_lump_cashflow,
    make_pure_dca_cashflow,
    make_retail_cashflow,
    metrics_dca,
    signal_ema,
)


# --------------------------------------------------------------------------
# Static mix backtest (no rotation, periodic rebalance)
# --------------------------------------------------------------------------

def backtest_static_mix(prices: dict[str, pd.Series],
                        weights: dict[str, float],
                        cashflow: CashflowPlan,
                        fee_bps: float = 2.5,
                        slip_bps: float = 5.0,
                        label: str = "Mix",
                        rebalance_freq: str = "Y") -> BacktestResult:
    """Static-weight portfolio with periodic rebalance + DCA cash injections.

    Daily mechanics:
        equity_t = equity_{t-1} * (1 + sum_i w_i * ret_{i,t}) - rebalance_cost
        equity_t += injection_t * (1 - buy_cost)   [cash flowing in]

    Where sum_i w_i = 1 and `weights` is constant *between* rebalance days; on
    rebalance days the implicit drift is reset to target weights and one-side
    cost is applied to the *traded* fraction (sum |w_target - w_drifted|).

    For simplicity we collapse this to a one-side cost on equity * sum |dw|,
    which is identical to selling the over-weight legs and buying the under-
    weight legs at one-side cost.

    `prices` keys must match `weights` keys. Values are price Series aligned
    on the same DatetimeIndex (load_universe already handles this for QQQ /
    SPY / AGG via reindex+ffill).
    """
    assert set(prices.keys()) == set(weights.keys()), "prices and weights must align"
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "weights must sum to 1"
    names = list(weights.keys())
    w_target = np.array([weights[n] for n in names], dtype=float)

    # Common index
    idx = None
    for s in prices.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.intersection(cashflow.injections.index).sort_values()

    rets = np.column_stack([prices[n].reindex(idx).pct_change().fillna(0.0).values
                             for n in names])
    inj = cashflow.injections.reindex(idx).fillna(0.0).astype(float).values
    cost_bps = (fee_bps + slip_bps) / 10000.0

    # Rebalance day mask: first business day each year (or month/quarter)
    if rebalance_freq == "Y":
        # First trading day of each calendar year (skip the very first day)
        rebalance_mask = pd.Series(False, index=idx)
        for y in pd.unique(idx.year):
            sel = idx[idx.year == y]
            rebalance_mask.loc[sel[0]] = True
        rebalance_mask.iloc[0] = False  # don't rebalance on day 0 (no drift yet)
    elif rebalance_freq == "Q":
        rebalance_mask = pd.Series(False, index=idx)
        seen = set()
        for ts in idx:
            key = (ts.year, ts.quarter)
            if key not in seen:
                rebalance_mask.loc[ts] = True
                seen.add(key)
        rebalance_mask.iloc[0] = False
    elif rebalance_freq == "M":
        rebalance_mask = pd.Series(False, index=idx)
        seen = set()
        for ts in idx:
            key = (ts.year, ts.month)
            if key not in seen:
                rebalance_mask.loc[ts] = True
                seen.add(key)
        rebalance_mask.iloc[0] = False
    else:
        raise ValueError(rebalance_freq)
    rb = rebalance_mask.values

    n = len(idx)
    n_assets = len(names)
    eq = np.zeros(n)
    held_dollars = np.zeros((n, n_assets))  # dollars per leg, post all events

    n_rebalances = 0
    for t in range(n):
        if t == 0:
            # Day 0: deposit cash, allocate to weights instantly (one-side buy cost)
            buy_cost = cost_bps if inj[t] > 0 else 0.0
            eq[0] = inj[0] * (1.0 - buy_cost)
            held_dollars[0] = eq[0] * w_target
        else:
            # 1) drift each leg by its return
            held_dollars[t] = held_dollars[t - 1] * (1.0 + rets[t])
            equity_drifted = held_dollars[t].sum()
            # 2) periodic rebalance: trade back to target weights at one-side cost
            if rb[t] and equity_drifted > 0:
                w_drifted = held_dollars[t] / equity_drifted
                turnover = float(np.abs(w_target - w_drifted).sum()) / 2.0
                # turnover here is the one-side dollar fraction traded
                rebalance_cost = equity_drifted * turnover * cost_bps
                equity_drifted -= rebalance_cost
                held_dollars[t] = equity_drifted * w_target
                n_rebalances += 1
            # 3) cash injection
            if inj[t] > 0:
                buy_cost = cost_bps  # always invested in static mix
                add = inj[t] * (1.0 - buy_cost)
                # New cash flows in at target weights
                held_dollars[t] += add * w_target
            eq[t] = held_dollars[t].sum()

    # held_asset for static mix is just "mix"
    held = np.array(["mix"] * n, dtype=object)

    cf = CashflowPlan(injections=cashflow.injections, label=label)
    return BacktestResult(
        equity=pd.Series(eq, index=idx),
        contributions_cum=pd.Series(inj, index=idx).cumsum(),
        held_asset=pd.Series(held, index=idx),
        n_rotations=n_rebalances,
        cashflow=cf,
        asset_on_name=names[0],
        asset_off_name=names[1] if n_assets > 1 else "",
        fee_bps=fee_bps,
        slip_bps=slip_bps,
    )


# --------------------------------------------------------------------------
# AGG synthesis: t-bill returns proxy for pre-2003-09-29
# --------------------------------------------------------------------------

def synthesize_agg(U: dict) -> pd.Series:
    """Stitch t-bill cumulative price (1999-2003-09-29) onto real AGG so that
    the 60/40 mix can run for the full 27-year window. Calibrate at splice."""
    qqq_idx = U["qqq"].index
    agg_real = U["agg"].dropna()
    if agg_real.empty:
        return U["tbill"].reindex(qqq_idx).ffill()
    splice_date = agg_real.index[0]
    tbill = U["tbill"]
    pre = tbill.loc[:splice_date].iloc[:-1]
    if len(pre) == 0:
        return agg_real.reindex(qqq_idx).ffill()
    calib = float(agg_real.iloc[0]) / float(tbill.loc[splice_date])
    pre_scaled = pre * calib
    full = pd.concat([pre_scaled, agg_real])
    return full.reindex(qqq_idx).ffill()


# --------------------------------------------------------------------------
# Build all baselines
# --------------------------------------------------------------------------

def run_baselines(initial: float, annual: float, U: dict | None = None,
                  contrib_freq: str = "M") -> dict[str, dict]:
    """Run all baselines + Retail Holy Grail under the same cashflow plan.

    Returns:
        {name: {'result': BacktestResult, 'metrics': dict}}
    """
    if U is None:
        U = load_universe()

    qqq = U["qqq"]
    tqqq = U["tqqq"]
    spy = U["spy"]
    agg = synthesize_agg(U)

    dates = qqq.index
    out: dict[str, dict] = {}

    # Shared cashflow for baselines 1–5 + Retail Holy Grail comparison
    cf_retail = make_retail_cashflow(initial, annual, dates, contrib_freq=contrib_freq)
    cf_pure_dca = make_pure_dca_cashflow(initial, annual, dates, contrib_freq=contrib_freq)
    cf_lump = make_lump_cashflow(initial, dates)

    # Build a reusable cashflow with a custom label per baseline
    def relabel(cf: CashflowPlan, name: str) -> CashflowPlan:
        return CashflowPlan(injections=cf.injections.copy(),
                            label=f"{name} ({cf.label})")

    # 1. DCA QQQ B&H — signal=0 → always OFF asset (QQQ)
    sig_zero = pd.Series(0.0, index=dates)
    cf1 = relabel(cf_retail, "DCA QQQ B&H")
    res1 = backtest_dca(qqq, qqq, sig_zero, cf1,
                        asset_on_name="QQQ", asset_off_name="QQQ")
    out["DCA QQQ B&H"] = {"result": res1, "metrics": metrics_dca(res1)}

    # 2. DCA SPY B&H
    cf2 = relabel(cf_retail, "DCA SPY B&H")
    res2 = backtest_dca(spy, spy, sig_zero, cf2,
                        asset_on_name="SPY", asset_off_name="SPY")
    out["DCA SPY B&H"] = {"result": res2, "metrics": metrics_dca(res2)}

    # 3. DCA TQQQ B&H — signal=1 → always ON asset (TQQQ)
    sig_one = pd.Series(1.0, index=dates)
    cf3 = relabel(cf_retail, "DCA TQQQ B&H")
    res3 = backtest_dca(tqqq, tqqq, sig_one, cf3,
                        asset_on_name="TQQQ", asset_off_name="TQQQ")
    out["DCA TQQQ B&H"] = {"result": res3, "metrics": metrics_dca(res3)}

    # 4. DCA 60/40 (QQQ + AGG)
    cf4 = relabel(cf_retail, "DCA 60/40 (QQQ + AGG)")
    res4 = backtest_static_mix(prices={"QQQ": qqq, "AGG": agg},
                               weights={"QQQ": 0.6, "AGG": 0.4},
                               cashflow=cf4,
                               label=cf4.label,
                               rebalance_freq="Y")
    out["DCA 60/40 (QQQ + AGG)"] = {"result": res4, "metrics": metrics_dca(res4)}

    # 5. DCA 60/40 (SPY + AGG)
    cf5 = relabel(cf_retail, "DCA 60/40 (SPY + AGG)")
    res5 = backtest_static_mix(prices={"SPY": spy, "AGG": agg},
                               weights={"SPY": 0.6, "AGG": 0.4},
                               cashflow=cf5,
                               label=cf5.label,
                               rebalance_freq="Y")
    out["DCA 60/40 (SPY + AGG)"] = {"result": res5, "metrics": metrics_dca(res5)}

    # 6. Pure DCA Rotation — EMA(5,200) rotation, 12-mo phase-in cashflow (no r-rule)
    sig_ema = signal_ema(qqq, 5, 200)
    cf6 = relabel(cf_pure_dca, "Pure DCA Rotation (no r-rule)")
    res6 = backtest_dca(tqqq, qqq, sig_ema, cf6,
                        asset_on_name="TQQQ", asset_off_name="QQQ")
    out["Pure DCA Rotation (no r-rule)"] = {"result": res6, "metrics": metrics_dca(res6)}

    # 7. Original Lump-sum Rotation — $10k LSI day 0, EMA(5,200) rotation
    cf7 = CashflowPlan(injections=cf_lump.injections.copy(),
                       label=f"Lump-sum Rotation EMA(5,200) ({cf_lump.label})")
    res7 = backtest_dca(tqqq, qqq, sig_ema, cf7,
                        asset_on_name="TQQQ", asset_off_name="QQQ")
    out["Lump-sum Rotation"] = {"result": res7, "metrics": metrics_dca(res7)}

    # 8. (Reference) Retail Holy Grail under the same cashflow as 1-5
    cf8 = relabel(cf_retail, "Retail Holy Grail (r-rule)")
    res8 = backtest_dca(tqqq, qqq, sig_ema, cf8,
                        asset_on_name="TQQQ", asset_off_name="QQQ")
    out["Retail Holy Grail"] = {"result": res8, "metrics": metrics_dca(res8)}

    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_summary(results: dict, sort_by: str = "xirr") -> None:
    """Pretty-print a sorted summary table."""
    rows = []
    for name, payload in results.items():
        m = payload["metrics"]
        rows.append({
            "name": name,
            "xirr": m["xirr"],
            "twr": m["twr"],
            "mdd_market": m["mdd_market"],
            "mdd_vs_contrib": m["mdd_vs_contrib"],
            "final_value": m["final_value"],
            "total_contributed": m["total_contributed"],
            "multiple": m["multiple"],
            "sharpe": m["sharpe"],
            "n_rotations": m["n_rotations"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    name_w = max(28, df["name"].str.len().max() + 2)

    print()
    width = name_w + 96
    print("=" * width)
    print(f"{'Baseline':<{name_w}}{'XIRR':>9}{'TWR':>9}{'MDD-mkt':>10}"
          f"{'MDD-contr':>12}{'Mult':>8}{'Final':>16}{'Sharpe':>9}{'Rot':>6}")
    print("-" * width)
    for _, r in df.iterrows():
        print(f"{r['name']:<{name_w}}{r['xirr']*100:>8.2f}%{r['twr']*100:>8.2f}%"
              f"{r['mdd_market']*100:>9.2f}%{r['mdd_vs_contrib']*100:>11.2f}%"
              f"{r['multiple']:>7.2f}x${r['final_value']:>14,.0f}"
              f"{r['sharpe']:>9.2f}{r['n_rotations']:>6}")
    print("=" * width)


def vs_retail_table(results: dict) -> None:
    """Print Retail Holy Grail vs each baseline (XIRR diff + multiple ratio)."""
    if "Retail Holy Grail" not in results:
        return
    rh = results["Retail Holy Grail"]["metrics"]
    print()
    print("=" * 100)
    print(f"{'Retail Holy Grail vs Baseline':<40}{'ΔXIRR':>10}"
          f"{'ΔFinal':>16}{'Mult ratio':>14}{'Verdict':>18}")
    print("-" * 100)
    for name, payload in results.items():
        if name == "Retail Holy Grail":
            continue
        m = payload["metrics"]
        d_xirr = rh["xirr"] - m["xirr"]
        d_final = rh["final_value"] - m["final_value"]
        ratio = rh["multiple"] / m["multiple"] if m["multiple"] > 0 else float("nan")
        verdict = "Retail wins" if d_xirr > 0 else "Baseline wins"
        print(f"{name:<40}{d_xirr*100:>9.2f}%${d_final:>14,.0f}{ratio:>13.2f}x{verdict:>18}")
    print("=" * 100)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial", type=float, default=10_000)
    ap.add_argument("--annual", type=float, default=10_000)
    ap.add_argument("--freq", choices=["M", "Q", "Y"], default="M")
    ap.add_argument("--out", type=str,
                    default="results/dca/baselines_results.pkl")
    args = ap.parse_args()

    print("[baselines] Loading universe ...")
    U = load_universe()
    print(f"  QQQ {U['qqq'].index[0].date()} -> {U['qqq'].index[-1].date()}"
          f" ({len(U['qqq'])} days)")
    print(f"  AGG real coverage: {U['agg'].dropna().index[0].date()} ->"
          f" {U['agg'].dropna().index[-1].date()}; pre-2003 synthesised from t-bill")
    print()

    print(f"[baselines] Running 7 baselines + Retail Holy Grail under "
          f"initial=${args.initial:,.0f} + ${args.annual:,.0f}/yr ({args.freq}) ...")
    results = run_baselines(args.initial, args.annual, U, contrib_freq=args.freq)

    # Per-baseline single-line metrics
    print()
    for name, payload in results.items():
        print(f"{name:<32} | {fmt_metric_row(payload['metrics'])}")

    # Sorted table
    print_summary(results, sort_by="xirr")

    # Verify lump-sum rotation reproduces README headline
    print()
    print("[verify] Lump-sum Rotation should match README headline "
          "(CAGR ~15.88%, Final ~$543,387):")
    lm = results["Lump-sum Rotation"]["metrics"]
    print(f"  XIRR  = {lm['xirr']*100:.2f}%")
    print(f"  TWR   = {lm['twr']*100:.2f}%")
    print(f"  Final = ${lm['final_value']:,.0f}")
    print(f"  MDD   = {lm['mdd_market']*100:.2f}%")
    print(f"  Sharpe= {lm['sharpe']:.2f}")
    target_cagr, target_final = 0.1588, 543_387
    cagr_ok = abs(lm["twr"] - target_cagr) < 0.005  # within 50 bps
    final_ok = abs(lm["final_value"] - target_final) / target_final < 0.05
    print(f"  CAGR within 50 bps of 15.88%? {'OK' if cagr_ok else 'MISS'}")
    print(f"  Final within 5%  of $543,387? {'OK' if final_ok else 'MISS'}")

    # Retail vs each baseline
    vs_retail_table(results)

    # Pickle
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "args": vars(args),
            "results": results,
        }, f)
    print()
    print(f"[baselines] Saved {len(results)} results to {out_path}")
    return results


if __name__ == "__main__":
    main()
