"""
Fixed-param strategies vs buy & hold — full CPCV comparison
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
import pickle

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0
FAST = 5
SLOW = 200
LEVELS = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]
EMBARGO = 21

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

def get_bull(close, fast, slow):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s)
    bull.iloc[:slow] = False
    return bull

def build_pyramid_pos(qqq_w, bull, ma_period, base_tqqq, levels, anchor):
    if anchor == "ma":
        ma = qqq_w.ewm(span=ma_period, adjust=False).mean()
        deviation = qqq_w / ma - 1
    else:
        peak = qqq_w.cummax()
        deviation = qqq_w / peak - 1
    tpos = pd.Series(0.0, index=qqq_w.index)
    for i in range(len(qqq_w.index)):
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

def cpcv_run(qqq_d, tqqq_d, n_splits, k_test, embargo, position_fn):
    """Generic CPCV: position_fn takes (qqq_window) → (qpos, tpos)"""
    n = len(qqq_d)
    fold_size = n // n_splits
    folds = [(i * fold_size, (i+1) * fold_size if i < n_splits-1 else n) for i in range(n_splits)]
    all_combos = list(combinations(range(n_splits), k_test))
    results = []
    for combo in all_combos:
        test_folds = list(combo)
        test_rets = []
        for tf in test_folds:
            tf_start, tf_end = folds[tf]
            tf_start_emb = tf_start + embargo
            tf_end_emb = max(tf_start_emb + 1, tf_end - embargo)
            if tf_end_emb - tf_start_emb < 100:
                continue
            test_idx = qqq_d.index[tf_start_emb:tf_end_emb]
            warm_start = max(0, tf_start_emb - SLOW - 50)
            full_idx = qqq_d.index[warm_start:tf_end_emb]
            qpos, tpos = position_fn(qqq_d.loc[full_idx])
            qpos_t = qpos.loc[test_idx]
            tpos_t = tpos.loc[test_idx]
            eq, ret = backtest_dual(qpos_t, tpos_t, qqq_d.loc[test_idx], tqqq_d.loc[test_idx])
            if not ret.empty:
                test_rets.append(ret)
        if test_rets:
            combined = pd.concat(test_rets)
            combined_eq = (1 + combined).cumprod() * INIT_CASH
            c, m, sh, ca, fv, so = metrics(combined_eq, combined)
            results.append({'cagr': c, 'mdd': m, 'sharpe': sh, 'calmar': ca, 'sortino': so})
    return results

# Strategy builders
def rotation_fixed(qqq_w):
    bull = get_bull(qqq_w, FAST, SLOW).astype(float)
    return 1 - bull, bull

def pyramid_t4_fixed(qqq_w):
    bull = get_bull(qqq_w, FAST, SLOW)
    tpos = build_pyramid_pos(qqq_w, bull, SLOW, 0.0, LEVELS, "peak")
    qpos = pd.Series(0.0, index=qqq_w.index)
    return qpos, tpos

def pyramid_t6_fixed(qqq_w):
    bull = get_bull(qqq_w, FAST, SLOW)
    tpos = build_pyramid_pos(qqq_w, bull, SLOW, 0.5, LEVELS, "ma")
    qpos = pd.Series(0.0, index=qqq_w.index)
    return qpos, tpos

def bh_qqq_fixed(qqq_w):
    qpos = pd.Series(1.0, index=qqq_w.index)
    tpos = pd.Series(0.0, index=qqq_w.index)
    return qpos, tpos

def bh_tqqq_fixed(qqq_w):
    qpos = pd.Series(0.0, index=qqq_w.index)
    tpos = pd.Series(1.0, index=qqq_w.index)
    return qpos, tpos

# ============================================================
# Full period single-path
# ============================================================
print("=" * 108)
print("FULL PERIOD 1999-2026 (single path)")
print("=" * 108)
print(f"\n{'Strategy':<40} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Final $10K':>14}")
print("-" * 108)

all_strats = [
    ("Rotation (fixed 5/200)", rotation_fixed),
    ("Pyramid T6 (fixed)", pyramid_t6_fixed),
    ("Pyramid T4 (fixed)", pyramid_t4_fixed),
    ("QQQ Buy & Hold", bh_qqq_fixed),
    ("TQQQ Buy & Hold (synth+real)", bh_tqqq_fixed),
]

for name, fn in all_strats:
    qpos, tpos = fn(qqq)
    eq, ret = backtest_dual(qpos, tpos, qqq, tqqq_full)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<40} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}")

# ============================================================
# CPCV comparison
# ============================================================
print()
print("=" * 108)
print(f"CPCV (N=10, k=2 → 45 paths) — fixed-param strategies vs buy & hold")
print("=" * 108)

cpcv_data = {}
for name, fn in all_strats:
    print(f"\nRunning CPCV: {name} ...")
    results = cpcv_run(qqq, tqqq_full, 10, 2, EMBARGO, fn)
    cpcv_data[name] = results

# Print summary
print(f"\n{'Strategy':<36} {'Med CAGR':>10} {'CAGR IQR':>22} {'Med Sharpe':>12} {'Med Calmar':>12} {'Med MDD':>10} {'P(>0)':>8} {'P(>15%)':>10}")
print("-" * 130)
for name, fn in all_strats:
    r = cpcv_data[name]
    cagrs = np.array([p['cagr'] for p in r])
    sharpes = np.array([p['sharpe'] for p in r])
    calmars = np.array([p['calmar'] for p in r])
    mdds = np.array([p['mdd'] for p in r])
    med_cagr = np.median(cagrs) * 100
    q25 = np.percentile(cagrs, 25) * 100
    q75 = np.percentile(cagrs, 75) * 100
    med_sh = np.median(sharpes)
    med_cal = np.median(calmars)
    med_mdd = np.median(mdds) * 100
    p_gt0 = (cagrs > 0).mean() * 100
    p_gt15 = (cagrs > 0.15).mean() * 100
    print(f"{name:<36} {med_cagr:>9.2f}% [{q25:>6.2f}%, {q75:>6.2f}%] {med_sh:>12.3f} {med_cal:>12.3f} {med_mdd:>9.2f}% {p_gt0:>7.1f}% {p_gt15:>9.1f}%")

# Save
with open("/tmp/ema_backtest/cpcv_vs_bh.pkl", "wb") as f:
    pickle.dump(cpcv_data, f)
