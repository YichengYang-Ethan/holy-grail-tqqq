"""
CPCV on fixed-parameter (non-adaptive) versions of top 3 strategies.
Goal: does removing the walk-forward parameter selection reduce noise
(fewer degrees of freedom) and improve CPCV stability?

Fixed parameters chosen a priori:
- EMA fast = 5, slow = 200 (the textbook MA 5/200 crossover)
- Pyramid levels: -10/-20/-30/-40 → 25/50/75/100%
- No parameter tuning at any step
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0
FAST = 5
SLOW = 200
LEVELS = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]

print("[1/4] Loading data ...")
qqq = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()

def build_synth(qqq_close, tqqq_real):
    qret = qqq_close.pct_change().fillna(0)
    expense_d = 0.0084 / 252
    def fin(d):
        y = d.year
        if y <= 2007: r = 0.045
        elif y <= 2008: r = 0.025
        elif y <= 2015: r = 0.0015
        elif y <= 2019: r = 0.015
        elif y <= 2021: r = 0.001
        else: r = 0.045
        return (r + 0.004) * 2 / 252
    fin_d = pd.Series([fin(d) for d in qqq_close.index], index=qqq_close.index)
    slip_d = 0.003 / 252
    synth_ret = 3 * qret - expense_d - fin_d - slip_d
    synth = (1 + synth_ret).cumprod()
    overlap = tqqq_real.index[0]
    calib = float(tqqq_real.iloc[0]) / float(synth.loc[overlap])
    pre = synth.loc[:overlap].iloc[:-1] * calib
    full = pd.concat([pre, tqqq_real])
    return full.reindex(qqq_close.index).ffill()

tqqq_full = build_synth(qqq, tqqq_real)
print(f"  Data: {qqq.index[0].date()} ~ {qqq.index[-1].date()}, {len(qqq)} days")
print(f"  FIXED params: fast={FAST}, slow={SLOW}, levels={LEVELS}")

def get_bull(close, fast, slow):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s)
    bull.iloc[:slow] = False
    return bull

def build_pyramid_pos(qqq_window, bull, ma_period, base_tqqq, levels, anchor):
    if anchor == "ma":
        ma = qqq_window.ewm(span=ma_period, adjust=False).mean()
        deviation = qqq_window / ma - 1
    else:
        peak = qqq_window.cummax()
        deviation = qqq_window / peak - 1
    tpos = pd.Series(0.0, index=qqq_window.index)
    for i in range(len(qqq_window.index)):
        if i < ma_period:
            tpos.iloc[i] = 0
            continue
        if bull.iloc[i]:
            tpos.iloc[i] = 1.0
        else:
            dev = deviation.iloc[i]
            deploy = 0.0
            for thresh, frac in levels:
                if dev <= thresh:
                    deploy = frac
            cash_reserve = 1.0 - base_tqqq
            tpos.iloc[i] = base_tqqq + cash_reserve * deploy
    return tpos

def backtest_dual(qpos, tpos, qqq_w, tqqq_w, fee_bps=2.5, slip_bps=5.0):
    qpos_l = qpos.shift(1).fillna(0)
    tpos_l = tpos.shift(1).fillna(0)
    qret = qqq_w.pct_change().fillna(0)
    tret = tqqq_w.pct_change().fillna(0)
    pos_change = qpos_l.diff().abs().fillna(0) + tpos_l.diff().abs().fillna(0)
    cost = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = qpos_l * qret + tpos_l * tret - cost
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret

def metrics(eq, ret):
    if eq.empty or eq.iloc[-1] <= 0:
        return -1, -1, 0, -1, 0, 0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0:
        return 0, 0, 0, 0, 0, 0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else cagr
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

# ============================================================
# Fixed-parameter strategy builders (NO adaptation)
# ============================================================
def strat_rotation_fixed(qqq_window):
    bull = get_bull(qqq_window, FAST, SLOW).astype(float)
    qpos = 1 - bull  # bear → QQQ
    tpos = bull      # bull → TQQQ
    return qpos, tpos

def strat_pyramid_t4_fixed(qqq_window):
    bull = get_bull(qqq_window, FAST, SLOW)
    tpos = build_pyramid_pos(qqq_window, bull, SLOW, 0.0, LEVELS, "peak")
    qpos = pd.Series(0.0, index=qqq_window.index)
    return qpos, tpos

def strat_pyramid_t6_fixed(qqq_window):
    bull = get_bull(qqq_window, FAST, SLOW)
    tpos = build_pyramid_pos(qqq_window, bull, SLOW, 0.5, LEVELS, "ma")
    qpos = pd.Series(0.0, index=qqq_window.index)
    return qpos, tpos

# ============================================================
# Single-path full-history backtest (baseline)
# ============================================================
print("\n[2/4] Single-path full-history backtest (no CPCV)")
print()

single_path_results = {}
for name, fn in [("Rotation (QQQ/TQQQ, fixed)", strat_rotation_fixed),
                  ("Pyramid T4 (peak, fixed)", strat_pyramid_t4_fixed),
                  ("Pyramid T6 (half-base MA, fixed)", strat_pyramid_t6_fixed)]:
    qpos, tpos = fn(qqq)
    eq, ret = backtest_dual(qpos, tpos, qqq, tqqq_full)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    single_path_results[name] = (c, m, sh, ca, fv, so, eq, ret)
    print(f"  {name:<40} CAGR {c*100:>6.2f}%  MDD {m*100:>7.2f}%  Sharpe {sh:.3f}  Calmar {ca:.3f}  Final ${fv:>10,.0f}")

# ============================================================
# CPCV on fixed-parameter strategies
# ============================================================
print("\n[3/4] CPCV (N=10 splits, k=2 test groups → 45 paths) on fixed strategies")
print()

EMBARGO = 21

def cpcv_fixed(qqq, tqqq, n_splits, k_test, embargo, strategy_fn):
    """CPCV with NO parameter tuning — strategy uses fixed params on every path"""
    n = len(qqq)
    fold_size = n // n_splits
    folds = [(i * fold_size, (i+1) * fold_size if i < n_splits-1 else n) for i in range(n_splits)]
    all_combos = list(combinations(range(n_splits), k_test))
    path_results = []
    for combo in all_combos:
        test_folds = list(combo)
        # Build test return series by running fixed strategy on each test fold
        test_returns = []
        for tf in test_folds:
            tf_start, tf_end = folds[tf]
            tf_start_emb = tf_start + embargo
            tf_end_emb = max(tf_start_emb + 1, tf_end - embargo)
            if tf_end_emb - tf_start_emb < 100:
                continue
            test_idx = qqq.index[tf_start_emb:tf_end_emb]
            # Need warm-up for EMA(200)
            warm_start = max(0, tf_start_emb - SLOW - 50)
            full_idx = qqq.index[warm_start:tf_end_emb]
            qpos_f, tpos_f = strategy_fn(qqq.loc[full_idx])
            qpos_test = qpos_f.loc[test_idx]
            tpos_test = tpos_f.loc[test_idx]
            eq, ret = backtest_dual(qpos_test, tpos_test, qqq.loc[test_idx], tqqq.loc[test_idx])
            if not ret.empty:
                test_returns.append(ret)
        if test_returns:
            combined = pd.concat(test_returns)
            combined_eq = (1 + combined).cumprod() * INIT_CASH
            c, m, sh, ca, fv, so = metrics(combined_eq, combined)
            path_results.append({'combo': combo, 'cagr': c, 'mdd': m,
                                 'sharpe': sh, 'calmar': ca, 'sortino': so})
    return path_results

cpcv_results = {}
for name, fn in [("Rotation (QQQ/TQQQ, fixed)", strat_rotation_fixed),
                  ("Pyramid T4 (peak, fixed)", strat_pyramid_t4_fixed),
                  ("Pyramid T6 (half-base MA, fixed)", strat_pyramid_t6_fixed)]:
    print(f"  Running CPCV: {name} ...")
    results = cpcv_fixed(qqq, tqqq_full, 10, 2, EMBARGO, fn)
    cpcv_results[name] = results
    cagrs = np.array([r['cagr'] for r in results])
    sharpes = np.array([r['sharpe'] for r in results])
    calmars = np.array([r['calmar'] for r in results])
    mdds = np.array([r['mdd'] for r in results])
    print(f"    {len(results)} valid paths")
    print(f"    CAGR   median {np.median(cagrs)*100:>6.2f}%  IQR [{np.percentile(cagrs, 25)*100:>6.2f}%, {np.percentile(cagrs, 75)*100:>6.2f}%]")
    print(f"    Sharpe median {np.median(sharpes):>6.3f}  IQR [{np.percentile(sharpes, 25):>6.3f}, {np.percentile(sharpes, 75):>6.3f}]")
    print(f"    Calmar median {np.median(calmars):>6.3f}  IQR [{np.percentile(calmars, 25):>6.3f}, {np.percentile(calmars, 75):>6.3f}]")
    print(f"    MDD    median {np.median(mdds)*100:>6.2f}%  worst {np.min(mdds)*100:>6.2f}%")
    print(f"    P(CAGR > 0): {(cagrs > 0).mean()*100:.1f}%")
    print(f"    P(CAGR > 15%): {(cagrs > 0.15).mean()*100:.1f}%")
    print()

# ============================================================
# Side-by-side comparison: fixed vs adaptive (from previous run)
# ============================================================
print("[4/4] Fixed-param vs adaptive-param CPCV comparison")
print()

# Adaptive (from previous walk-forward CPCV)
adaptive_summary = {
    "Rotation (adaptive WF)": {"median_cagr": 5.24, "iqr_cagr": "[-1.81%, 13.71%]",
                                  "median_calmar": 0.079, "p_gt_15": 24.4},
    "Pyramid T4 (adaptive WF)": {"median_cagr": 2.99, "iqr_cagr": "[-9.30%, 14.43%]",
                                     "median_calmar": 0.058, "p_gt_15": 24.4},
    "Pyramid T6 (adaptive WF)": {"median_cagr": 4.83, "iqr_cagr": "[-3.37%, 14.81%]",
                                    "median_calmar": 0.085, "p_gt_15": 24.4},
}

print(f"  {'Strategy':<36} {'Median CAGR':>13} {'CAGR IQR':>22} {'Median Calmar':>16} {'P(>15%)':>10}")
print("  " + "-" * 104)
print("  FIXED PARAMETERS (no walk-forward selection):")
for name, results in cpcv_results.items():
    cagrs = np.array([r['cagr'] for r in results])
    calmars = np.array([r['calmar'] for r in results])
    q25 = np.percentile(cagrs, 25) * 100
    q75 = np.percentile(cagrs, 75) * 100
    iqr = f"[{q25:>6.2f}%, {q75:>6.2f}%]"
    print(f"  {name:<36} {np.median(cagrs)*100:>12.2f}% {iqr:>22} {np.median(calmars):>16.3f} {(cagrs > 0.15).mean()*100:>9.1f}%")

print("\n  ADAPTIVE WALK-FORWARD (previous analysis):")
for name, s in adaptive_summary.items():
    print(f"  {name:<36} {s['median_cagr']:>12.2f}% {s['iqr_cagr']:>22} {s['median_calmar']:>16.3f} {s['p_gt_15']:>9.1f}%")

print()
print("=" * 100)
print("KEY QUESTIONS ANSWERED:")
print("  Q: Does removing adaptive tuning (fewer degrees of freedom) tighten CPCV distribution?")
print("  A: Compare the IQR widths above. Smaller IQR = more stable edge.")
print()
print("  Q: Does fixed-param median CAGR beat or trail adaptive?")
print("  A: Compare medians above.")

# Save
import pickle
with open("/tmp/ema_backtest/fixed_cpcv_results.pkl", "wb") as f:
    pickle.dump({'single_path': single_path_results, 'cpcv': cpcv_results}, f)
