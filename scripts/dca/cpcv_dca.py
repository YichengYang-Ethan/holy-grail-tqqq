"""
CPCV adapted to the DCA cashflow model (retail Holy Grail vs DCA QQQ B&H).

Mirrors the methodology in `scripts/fixed_params_cpcv.py` (CPCV with N=10,
k=2 -> 45 paths, embargo=21) and `scripts/forward_wf_proper.py` (causal
walk-forward, train pre-2010 then test 2010-2026), but replaces the
lump-sum return-compounding model with the DCA share-tracking engine in
`scripts/dca/core.py`.

Three deliverables:
  1. ROLLING-WINDOW CPCV (main output) -- monthly start dates over the
     full ~27-year span, fixed-length window of 5/10/15 years. Each
     window simulates the retail strategy AND a DCA-only QQQ B&H baseline
     with identical cashflow. Reports the distribution of XIRR / TWR /
     MDD / outperformance across all windows.

  2. FOLD-BASED CPCV (robustness check) -- the original Lopez de Prado
     CPCV, N=10 folds with k=2 test folds = 45 path-combinations. Within
     each test fold, run a fresh retail DCA backtest (initial+annual
     applied to that fold's timeline; the cashflow is therefore scaled
     by fold duration via the natural r-rule).

  3. CAUSAL WALK-FORWARD: train on pre-2010 data (fix params via the
     textbook EMA(5,200) rotation -- the same fixed-params choice as
     `fixed_params_cpcv.py`), test on 2010-2026 out-of-sample, retail
     DCA cashflow ($10K initial + $10K/yr).

Outputs:
  - `results/dca/cpcv_dca_results.pkl`  with keys:
      rolling_5yr, rolling_10yr, rolling_15yr  (DataFrames, one row per window)
      fold_cpcv_45paths                         (DataFrame, one row per fold/test)
      wf_oos                                   (dict)

The script ABSOLUTELY DOES NOT MUTATE core.py / retail_holy_grail.py.
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from scripts.dca.core import (
    BacktestResult,
    CashflowPlan,
    backtest_dca,
    load_universe,
    make_retail_cashflow,
    metrics_dca,
    signal_ema,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------------
# Defaults (mirror fixed_params_cpcv.py + retail_holy_grail.py)
# ----------------------------------------------------------------------------
FAST = 5
SLOW = 200
EMBARGO = 21          # 21-day embargo per Lopez de Prado
N_SPLITS = 10
K_TEST = 2            # C(10, 2) = 45 paths
INITIAL = 10_000.0
ANNUAL = 10_000.0
WF_TRAIN_END = "2010-02-10"

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "dca",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _slice_universe(U: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Return a per-window universe slice. The signal is computed on the
    sliced QQQ so EMA warm-up uses only in-window history (consistent with
    the assumption that a real retail investor starts from this date)."""
    mask = (U["qqq"].index >= start) & (U["qqq"].index <= end)
    idx = U["qqq"].index[mask]
    return {
        "qqq": U["qqq"].reindex(idx),
        "tqqq": U["tqqq"].reindex(idx),
    }


def _retail_metrics_on_window(initial: float, annual: float,
                              qqq: pd.Series, tqqq: pd.Series,
                              fast: int = FAST, slow: int = SLOW) -> dict:
    sig = signal_ema(qqq, fast, slow)
    cf = make_retail_cashflow(initial, annual, qqq.index, contrib_freq="M")
    res = backtest_dca(tqqq, qqq, sig, cf,
                        asset_on_name="TQQQ", asset_off_name="QQQ")
    return metrics_dca(res)


def _qqq_dca_metrics_on_window(initial: float, annual: float,
                                qqq: pd.Series) -> dict:
    """DCA into QQQ buy & hold using the same retail cashflow plan
    (so it's a fair comparison: same cash injected at same dates)."""
    cf = make_retail_cashflow(initial, annual, qqq.index, contrib_freq="M")
    # Always-on signal -> stays in price_on (=qqq) the whole time.
    sig = pd.Series(1.0, index=qqq.index)
    sig.iloc[0] = 1.0  # immediately on
    res = backtest_dca(qqq, qqq, sig, cf,
                        asset_on_name="QQQ", asset_off_name="QQQ")
    m = metrics_dca(res)
    m["label"] = "DCA QQQ B&H"
    return m


# ----------------------------------------------------------------------------
# 1. Rolling-window CPCV (main output)
# ----------------------------------------------------------------------------

def rolling_window_cpcv(initial: float, annual: float, window_years: int,
                         step_months: int = 1, U: Optional[dict] = None) -> pd.DataFrame:
    """Slide a fixed-length window across the full QQQ history. For every
    starting date (first business day of each month) where a full
    `window_years` window fits, run BOTH the retail strategy and the DCA
    QQQ B&H baseline using identical cashflow.

    Returns a DataFrame, one row per window, with columns:
        start, end, window_years,
        retail_xirr, retail_twr, retail_final, retail_mdd, retail_multiple,
        qqq_xirr, qqq_twr, qqq_final, qqq_mdd, qqq_multiple,
        retail_minus_qqq_xirr, retail_beats_qqq, retail_xirr_gt_15
    """
    if U is None:
        U = load_universe()
    qqq = U["qqq"]
    tqqq = U["tqqq"]
    full_idx = qqq.index

    # Candidate start dates = first business day of each month
    df = pd.DataFrame({"d": full_idx}, index=full_idx)
    fbom = pd.DatetimeIndex(df.groupby([full_idx.year, full_idx.month]).first()["d"].values)
    fbom = fbom.sort_values()
    if step_months != 1:
        fbom = fbom[::step_months]

    rows = []
    for start in fbom:
        end_target = start + pd.DateOffset(years=window_years)
        # Need at least window_years of history; pick last in-range date
        in_range = full_idx[(full_idx >= start) & (full_idx <= end_target)]
        if len(in_range) < int(window_years * 230):  # ~230 trading days/yr threshold
            continue
        win_qqq = qqq.loc[in_range]
        win_tqqq = tqqq.loc[in_range]
        try:
            r_m = _retail_metrics_on_window(initial, annual, win_qqq, win_tqqq)
            q_m = _qqq_dca_metrics_on_window(initial, annual, win_qqq)
        except Exception:
            continue
        delta = r_m["xirr"] - q_m["xirr"] if (
            not np.isnan(r_m["xirr"]) and not np.isnan(q_m["xirr"])) else float("nan")
        rows.append({
            "start": start,
            "end": in_range[-1],
            "window_years": window_years,
            "retail_xirr": r_m["xirr"],
            "retail_twr": r_m["twr"],
            "retail_final": r_m["final_value"],
            "retail_multiple": r_m["multiple"],
            "retail_mdd_market": r_m["mdd_market"],
            "retail_mdd_vs_contrib": r_m["mdd_vs_contrib"],
            "retail_n_rotations": r_m["n_rotations"],
            "qqq_xirr": q_m["xirr"],
            "qqq_twr": q_m["twr"],
            "qqq_final": q_m["final_value"],
            "qqq_multiple": q_m["multiple"],
            "qqq_mdd_market": q_m["mdd_market"],
            "qqq_mdd_vs_contrib": q_m["mdd_vs_contrib"],
            "retail_minus_qqq_xirr": delta,
            "retail_beats_qqq": (
                bool(delta > 0) if not np.isnan(delta) else False),
            "retail_xirr_gt_15": (
                bool(r_m["xirr"] > 0.15) if not np.isnan(r_m["xirr"]) else False),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. Fold-based CPCV with DCA cashflow (robustness check)
# ----------------------------------------------------------------------------

def fold_based_cpcv_dca(initial: float, annual: float, U: dict,
                         n_splits: int = N_SPLITS,
                         k_test: int = K_TEST,
                         embargo: int = EMBARGO) -> pd.DataFrame:
    """Original CPCV layout (10 folds, k=2 test -> 45 path combinations) with
    DCA cashflow. Within each test fold:
      - take dates [fold_start + embargo, fold_end - embargo]
      - run retail DCA + DCA QQQ B&H on that calendar
      - cashflow is the natural retail (initial, annual) applied to the
        fold's timeline -- this scales contributions to fold duration
        because annual is a per-year rate.

    Returns one row per (path, fold) so 45 * (~2 folds) = ~90 rows max
    (some excluded if fold is too short after embargo).
    """
    qqq = U["qqq"]
    tqqq = U["tqqq"]
    n = len(qqq)
    fold_size = n // n_splits
    folds = [(i * fold_size,
              (i + 1) * fold_size if i < n_splits - 1 else n)
             for i in range(n_splits)]

    all_combos = list(combinations(range(n_splits), k_test))
    rows = []
    for combo_id, combo in enumerate(all_combos):
        for tf in combo:
            tf_start, tf_end = folds[tf]
            tf_start_emb = tf_start + embargo
            tf_end_emb = max(tf_start_emb + 1, tf_end - embargo)
            if tf_end_emb - tf_start_emb < SLOW + 60:
                # Need EMA(slow=200) warm-up + ~60 days of usable signal
                continue
            test_idx = qqq.index[tf_start_emb:tf_end_emb]
            win_qqq = qqq.loc[test_idx]
            win_tqqq = tqqq.loc[test_idx]
            try:
                r_m = _retail_metrics_on_window(initial, annual, win_qqq, win_tqqq)
                q_m = _qqq_dca_metrics_on_window(initial, annual, win_qqq)
            except Exception:
                continue
            delta = r_m["xirr"] - q_m["xirr"] if (
                not np.isnan(r_m["xirr"]) and not np.isnan(q_m["xirr"])) else float("nan")
            rows.append({
                "combo_id": combo_id,
                "combo": combo,
                "fold": tf,
                "fold_start": test_idx[0],
                "fold_end": test_idx[-1],
                "fold_days": len(test_idx),
                "retail_xirr": r_m["xirr"],
                "retail_twr": r_m["twr"],
                "retail_final": r_m["final_value"],
                "retail_mdd_market": r_m["mdd_market"],
                "qqq_xirr": q_m["xirr"],
                "qqq_twr": q_m["twr"],
                "qqq_final": q_m["final_value"],
                "qqq_mdd_market": q_m["mdd_market"],
                "retail_minus_qqq_xirr": delta,
                "retail_beats_qqq": (
                    bool(delta > 0) if not np.isnan(delta) else False),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Causal walk-forward: pre-2010 train -> 2010-2026 OOS test
# ----------------------------------------------------------------------------

def walk_forward_dca(U: dict, train_end: str = WF_TRAIN_END,
                      initial: float = INITIAL,
                      annual: float = ANNUAL) -> dict:
    """Causal walk-forward: 'train' on pre-2010 data (we keep the textbook
    EMA(5, 200) main strategy fixed -- consistent with `fixed_params_cpcv.py`,
    so the train phase mostly verifies that 5/200 worked in-sample), then
    test out-of-sample on 2010-2026 with the retail DCA cashflow.

    Returns a dict with both the in-sample (train) and out-of-sample (test)
    metrics, plus a comparison vs DCA QQQ B&H on the test span.
    """
    qqq = U["qqq"]
    tqqq = U["tqqq"]
    train_end_ts = pd.Timestamp(train_end)

    # -- Train phase --------------------------------------------------------
    train_qqq = qqq.loc[:train_end_ts]
    train_tqqq = tqqq.loc[:train_end_ts]
    train_metrics = _retail_metrics_on_window(initial, annual, train_qqq, train_tqqq)

    # -- Test phase (OOS) ---------------------------------------------------
    # Skip embargo days after train_end
    train_pos = qqq.index.searchsorted(train_end_ts, side="right")
    test_start_idx = min(train_pos + EMBARGO, len(qqq) - 1)
    test_idx = qqq.index[test_start_idx:]
    test_qqq = qqq.loc[test_idx]
    test_tqqq = tqqq.loc[test_idx]
    test_metrics = _retail_metrics_on_window(initial, annual, test_qqq, test_tqqq)
    qqq_oos_metrics = _qqq_dca_metrics_on_window(initial, annual, test_qqq)

    return {
        "fast": FAST,
        "slow": SLOW,
        "train_end": train_end,
        "train_span": (train_qqq.index[0], train_qqq.index[-1]),
        "test_span": (test_qqq.index[0], test_qqq.index[-1]),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "qqq_test_metrics": qqq_oos_metrics,
        "test_minus_qqq_xirr": (
            test_metrics["xirr"] - qqq_oos_metrics["xirr"]
            if not np.isnan(test_metrics["xirr"])
            and not np.isnan(qqq_oos_metrics["xirr"]) else float("nan")),
    }


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------

def _summarize_rolling(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"{label}: no windows."
    rx = df["retail_xirr"].dropna()
    qx = df["qqq_xirr"].dropna()
    delta = df["retail_minus_qqq_xirr"].dropna()
    p_beat = df["retail_beats_qqq"].mean() * 100 if len(df) else float("nan")
    p_gt15 = df["retail_xirr_gt_15"].mean() * 100 if len(df) else float("nan")

    def pct_iqr(s):
        if s.empty:
            return "(empty)"
        return (f"P25={np.percentile(s, 25)*100:.2f}% "
                f"P50={np.percentile(s, 50)*100:.2f}% "
                f"P75={np.percentile(s, 75)*100:.2f}%")

    return (f"{label}  N={len(df)}\n"
            f"  retail XIRR: {pct_iqr(rx)}\n"
            f"  QQQ DCA XIRR: {pct_iqr(qx)}\n"
            f"  retail-QQQ delta: {pct_iqr(delta)}\n"
            f"  P(retail > QQQ B&H) = {p_beat:.1f}%   "
            f"P(retail XIRR > 15%) = {p_gt15:.1f}%")


def _summarize_fold(df: pd.DataFrame) -> str:
    if df.empty:
        return "fold-based CPCV: no valid folds."
    rx = df["retail_xirr"].dropna()
    qx = df["qqq_xirr"].dropna()
    delta = df["retail_minus_qqq_xirr"].dropna()
    p_beat = df["retail_beats_qqq"].mean() * 100 if len(df) else float("nan")

    def pct_iqr(s):
        if s.empty:
            return "(empty)"
        return (f"P25={np.percentile(s, 25)*100:.2f}% "
                f"P50={np.percentile(s, 50)*100:.2f}% "
                f"P75={np.percentile(s, 75)*100:.2f}%")

    return (f"fold-based CPCV (45 paths, {len(df)} folds total)\n"
            f"  retail XIRR: {pct_iqr(rx)}\n"
            f"  QQQ DCA XIRR: {pct_iqr(qx)}\n"
            f"  retail-QQQ delta: {pct_iqr(delta)}\n"
            f"  P(retail > QQQ B&H) = {p_beat:.1f}%")


def _summarize_wf(wf: dict) -> str:
    tr = wf["train_metrics"]
    te = wf["test_metrics"]
    qq = wf["qqq_test_metrics"]
    return ("Walk-forward (causal, EMA(5,200) fixed):\n"
            f"  Train {wf['train_span'][0].date()}->{wf['train_span'][1].date()}: "
            f"XIRR {tr['xirr']*100:.2f}%, MDD {tr['mdd_market']*100:.2f}%, "
            f"final ${tr['final_value']:,.0f}\n"
            f"  Test  {wf['test_span'][0].date()}->{wf['test_span'][1].date()}: "
            f"XIRR {te['xirr']*100:.2f}%, MDD {te['mdd_market']*100:.2f}%, "
            f"final ${te['final_value']:,.0f}\n"
            f"  QQQ DCA test: XIRR {qq['xirr']*100:.2f}%, "
            f"MDD {qq['mdd_market']*100:.2f}%, final ${qq['final_value']:,.0f}\n"
            f"  retail-QQQ XIRR delta (OOS): "
            f"{wf['test_minus_qqq_xirr']*100:.2f}%")


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------

def main():
    print("[cpcv_dca] Loading universe (one-shot, no per-window re-download) ...")
    U = load_universe()
    print(f"  QQQ {U['qqq'].index[0].date()} -> {U['qqq'].index[-1].date()} "
          f"({len(U['qqq'])} days)")
    print()

    # --- 1. Rolling-window CPCV (main output) ------------------------------
    rolling_results = {}
    for win_y in [5, 10, 15]:
        print(f"[1/3] Rolling-window CPCV (window = {win_y}y, monthly steps) ...")
        df = rolling_window_cpcv(INITIAL, ANNUAL, win_y, step_months=1, U=U)
        print(f"      -> {len(df)} valid {win_y}y windows.")
        rolling_results[f"rolling_{win_y}yr"] = df

    # --- 2. Fold-based CPCV ------------------------------------------------
    print()
    print(f"[2/3] Fold-based CPCV (N={N_SPLITS}, k={K_TEST} -> "
          f"{len(list(combinations(range(N_SPLITS), K_TEST)))} paths, "
          f"embargo={EMBARGO}d) ...")
    fold_df = fold_based_cpcv_dca(INITIAL, ANNUAL, U,
                                    n_splits=N_SPLITS,
                                    k_test=K_TEST,
                                    embargo=EMBARGO)
    print(f"      -> {len(fold_df)} valid (path, fold) rows.")

    # --- 3. Walk-forward (causal) ------------------------------------------
    print()
    print(f"[3/3] Walk-forward (train pre-{WF_TRAIN_END}, OOS test 2010+) ...")
    wf = walk_forward_dca(U, train_end=WF_TRAIN_END,
                            initial=INITIAL, annual=ANNUAL)
    print("      -> done.")

    # --- Persist -----------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "cpcv_dca_results.pkl")
    payload = {
        **rolling_results,
        "fold_cpcv_45paths": fold_df,
        "wf_oos": wf,
        "config": {
            "initial": INITIAL,
            "annual": ANNUAL,
            "fast": FAST,
            "slow": SLOW,
            "n_splits": N_SPLITS,
            "k_test": K_TEST,
            "embargo": EMBARGO,
            "wf_train_end": WF_TRAIN_END,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved: {out_path}")

    # --- Summary report ----------------------------------------------------
    print()
    print("=" * 100)
    print("CPCV DCA SUMMARY (initial = $10K, annual = $10K/yr, EMA 5/200)")
    print("=" * 100)
    print()
    for win_y in [5, 10, 15]:
        print(_summarize_rolling(rolling_results[f"rolling_{win_y}yr"],
                                   f"rolling-{win_y}yr"))
        print()
    print(_summarize_fold(fold_df))
    print()
    print(_summarize_wf(wf))
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
