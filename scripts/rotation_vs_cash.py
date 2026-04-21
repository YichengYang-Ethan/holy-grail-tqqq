"""
Direct comparison: Rotation (QQQ/TQQQ) vs Cash/TQQQ
Both fixed EMA 5/200, same data, same costs, full 1999-2026 + CPCV
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

INIT = 10_000.0
EMBARGO = 21
FAST, SLOW = 5, 200

qqq = yf.download("QQQ", start="1999-03-10", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()

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

def backtest_dual(qpos, tpos, qqq_w, tqqq_w, fee_bps=2.5, slip_bps=5.0):
    qpos_l = qpos.shift(1).fillna(0)
    tpos_l = tpos.shift(1).fillna(0)
    qret = qqq_w.pct_change().fillna(0)
    tret = tqqq_w.pct_change().fillna(0)
    pos_change = qpos_l.diff().abs().fillna(0) + tpos_l.diff().abs().fillna(0)
    cost = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = qpos_l * qret + tpos_l * tret - cost
    eq = (1 + strat_ret).cumprod() * INIT
    return eq, strat_ret

def metrics(eq):
    if eq.empty or eq.iloc[-1] <= 0: return -1, -1, 0, -1, 0, 0
    ret = eq.pct_change().fillna(0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else cagr
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

# Strategy builders
def rotation(qqq_w):
    bull = get_bull(qqq_w, FAST, SLOW).astype(float)
    return 1 - bull, bull  # bear→QQQ, bull→TQQQ

def cash_tqqq(qqq_w):
    bull = get_bull(qqq_w, FAST, SLOW).astype(float)
    return pd.Series(0.0, index=qqq_w.index), bull  # bear→cash, bull→TQQQ

# ============================================================
# FULL-PERIOD SINGLE PATH
# ============================================================
print("=" * 95)
print("SINGLE PATH 1999-2026")
print("=" * 95)
print(f"\n{'Strategy':<40} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Final $10K':>13}")
print("-" * 95)

qpos_r, tpos_r = rotation(qqq)
eq_rot, _ = backtest_dual(qpos_r, tpos_r, qqq, tqqq_full)
c, m, sh, ca, fv, so = metrics(eq_rot)
print(f"{'Rotation (QQQ/TQQQ, fixed 5/200)':<40} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>13,.0f}")

qpos_c, tpos_c = cash_tqqq(qqq)
eq_cash, _ = backtest_dual(qpos_c, tpos_c, qqq, tqqq_full)
c, m, sh, ca, fv, so = metrics(eq_cash)
print(f"{'TQQQ/Cash (pure, fixed 5/200)':<40} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>13,.0f}")

# ============================================================
# POST-2010 (Real TQQQ only)
# ============================================================
print()
print("=" * 95)
print("POST-2010 ONLY (real TQQQ data, no synthetic)")
print("=" * 95)
print(f"\n{'Strategy':<40} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Final $10K':>13}")
print("-" * 95)

qqq_2010 = qqq.loc["2010-02-11":]
tqqq_2010 = tqqq_full.loc["2010-02-11":]

qpos_r, tpos_r = rotation(qqq_2010)
eq_rot_2010, _ = backtest_dual(qpos_r, tpos_r, qqq_2010, tqqq_2010)
c, m, sh, ca, fv, so = metrics(eq_rot_2010)
print(f"{'Rotation (QQQ/TQQQ)':<40} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>13,.0f}")

qpos_c, tpos_c = cash_tqqq(qqq_2010)
eq_cash_2010, _ = backtest_dual(qpos_c, tpos_c, qqq_2010, tqqq_2010)
c, m, sh, ca, fv, so = metrics(eq_cash_2010)
print(f"{'TQQQ/Cash (pure)':<40} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>13,.0f}")

# ============================================================
# CPCV
# ============================================================
print()
print("=" * 95)
print("CPCV 45 PATHS (1999-2026)")
print("=" * 95)

def cpcv_run(qqq_d, tqqq_d, n_splits, k_test, embargo, strategy_fn):
    n = len(qqq_d)
    fold_size = n // n_splits
    folds = [(i * fold_size, (i+1) * fold_size if i < n_splits-1 else n) for i in range(n_splits)]
    all_combos = list(combinations(range(n_splits), k_test))
    path_results = []
    for combo in all_combos:
        test_rets = []
        for tf in combo:
            tf_start, tf_end = folds[tf]
            tf_start_emb = tf_start + embargo
            tf_end_emb = max(tf_start_emb + 1, tf_end - embargo)
            if tf_end_emb - tf_start_emb < 100: continue
            test_idx = qqq_d.index[tf_start_emb:tf_end_emb]
            warm_start = max(0, tf_start_emb - SLOW - 50)
            full_idx = qqq_d.index[warm_start:tf_end_emb]
            qpos_f, tpos_f = strategy_fn(qqq_d.loc[full_idx])
            qpos_test = qpos_f.loc[test_idx]
            tpos_test = tpos_f.loc[test_idx]
            eq, ret = backtest_dual(qpos_test, tpos_test, qqq_d.loc[test_idx], tqqq_d.loc[test_idx])
            if not ret.empty:
                test_rets.append(ret)
        if test_rets:
            combined = pd.concat(test_rets)
            combined_eq = (1 + combined).cumprod() * INIT
            c, m, sh, ca, fv, so = metrics(combined_eq)
            path_results.append({'cagr': c, 'mdd': m, 'sharpe': sh, 'calmar': ca, 'sortino': so})
    return path_results

print(f"\n{'Strategy':<35} {'Med CAGR':>10} {'CAGR IQR':>22} {'Med Sharpe':>11} {'Med Calmar':>11} {'Med MDD':>10} {'P(>0)':>7} {'P(>15%)':>9}")
print("-" * 125)

for name, fn in [("Rotation (QQQ/TQQQ, fixed)", rotation),
                 ("TQQQ/Cash (pure, fixed)", cash_tqqq)]:
    results = cpcv_run(qqq, tqqq_full, 10, 2, EMBARGO, fn)
    if not results: continue
    cagrs = np.array([p['cagr'] for p in results])
    sharpes = np.array([p['sharpe'] for p in results])
    calmars = np.array([p['calmar'] for p in results])
    mdds = np.array([p['mdd'] for p in results])
    q25 = np.percentile(cagrs, 25) * 100
    q75 = np.percentile(cagrs, 75) * 100
    iqr = f"[{q25:>6.2f}%, {q75:>6.2f}%]"
    print(f"{name:<35} {np.median(cagrs)*100:>9.2f}% {iqr:>22} {np.median(sharpes):>11.3f} {np.median(calmars):>11.3f} {np.median(mdds)*100:>9.2f}% {(cagrs>0).mean()*100:>6.1f}% {(cagrs>0.15).mean()*100:>8.1f}%")
