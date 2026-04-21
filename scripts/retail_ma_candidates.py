"""
Find simple MA-based strategies that beat QQQ Buy & Hold on ALL metrics:
- Higher CAGR
- Lower MDD
- Higher Sharpe
- Higher Calmar
- Retail-friendly (simple, infrequent trading, no options)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

INIT = 10_000.0
EMBARGO = 21
FAST, SLOW = 5, 200

print("Loading data ...")
qqq = yf.download("QQQ", start="1999-03-10", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
spy = yf.download("SPY", start="1999-03-10", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
tlt = yf.download("TLT", start="2002-07-30", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
agg = yf.download("AGG", start="2003-09-29", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
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

# Common period where AGG available (2003+)
common_start = pd.Timestamp("2003-10-01")
qqq_c = qqq.loc[common_start:]
spy_c = spy.loc[common_start:]
tqqq_c = tqqq_full.loc[common_start:]
tlt_c = tlt.reindex(qqq_c.index).ffill()
agg_c = agg.reindex(qqq_c.index).ffill()

print(f"Test period: {qqq_c.index[0].date()} ~ {qqq_c.index[-1].date()}")

def get_bull(close, fast, slow):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s)
    bull.iloc[:slow] = False
    return bull

def metrics(eq):
    if eq.empty or eq.iloc[-1] <= 0: return -1, -1, 0, -1, 0, 0
    ret = eq.pct_change().fillna(0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0: return 0, 0, 0, 0, 0, 0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else cagr
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

def backtest_rotation(risk_on_px, risk_off_px, signal_px, fee_bps=2.5, slip_bps=5.0):
    """Generic bull/bear rotation"""
    bull = get_bull(signal_px, FAST, SLOW).astype(float)
    on_pos = bull
    off_pos = 1 - bull
    on_ret = risk_on_px.pct_change().fillna(0)
    off_ret = risk_off_px.pct_change().fillna(0)
    pc = on_pos.diff().abs().fillna(0) + off_pos.diff().abs().fillna(0)
    cost = pc * (fee_bps + slip_bps) / 10000
    strat_ret = on_pos.shift(1).fillna(0) * on_ret + off_pos.shift(1).fillna(0) * off_ret - cost
    eq = (1 + strat_ret).cumprod() * INIT
    return eq

# ============================================================
print(f"\n{'=' * 108}")
print("CANDIDATE STRATEGIES — EMA 5/200 rotations")
print("=" * 108)

candidates = {
    # Unleveraged (retail-friendly)
    "QQQ/Cash rotation (EMA 5/200)":    lambda: backtest_rotation(qqq_c, pd.Series(1.0, index=qqq_c.index).cumprod(), qqq_c),
    "QQQ/TLT rotation":                  lambda: backtest_rotation(qqq_c, tlt_c, qqq_c),
    "QQQ/AGG rotation":                  lambda: backtest_rotation(qqq_c, agg_c, qqq_c),
    "SPY/Cash rotation":                 lambda: backtest_rotation(spy_c, pd.Series(1.0, index=spy_c.index).cumprod(), spy_c),
    "SPY/TLT rotation":                  lambda: backtest_rotation(spy_c, tlt_c, spy_c),
    "SPY/AGG rotation":                  lambda: backtest_rotation(spy_c, agg_c, spy_c),
    # Mild leverage
    "QQQ→TQQQ/AGG rotation":            lambda: backtest_rotation(tqqq_c, agg_c, qqq_c),
    "QQQ→TQQQ/TLT rotation":            lambda: backtest_rotation(tqqq_c, tlt_c, qqq_c),
    "QQQ→TQQQ/QQQ rotation (baseline)": lambda: backtest_rotation(tqqq_c, qqq_c, qqq_c),
}

print(f"\n{'Strategy':<42} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Final $10K':>14}")
print("-" * 108)

results = {}
for name, fn in candidates.items():
    eq = fn()
    c, m, sh, ca, fv, so = metrics(eq)
    results[name] = (c, m, sh, ca, fv, so, eq)
    mdd_ok = "✅" if m > -0.50 else ("⚠️" if m > -0.75 else "❌")
    print(f"{name:<42} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f} {mdd_ok}")

# Reference: QQQ B&H on same period
qqq_bh = (qqq_c / qqq_c.iloc[0]) * INIT
c, m, sh, ca, fv, so = metrics(qqq_bh)
print(f"{'★ QQQ Buy & Hold (benchmark)':<42} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}")

bench_cagr, bench_mdd, bench_sh, bench_cal = c, m, sh, ca

# ============================================================
# Check which strategies beat QQQ B&H on ALL metrics
# ============================================================
print(f"\n{'=' * 108}")
print(f"STRATEGIES THAT BEAT QQQ B&H ON ALL 4 METRICS (CAGR↑, MDD↑, Sharpe↑, Calmar↑):")
print("=" * 108)

print(f"\nBenchmark: CAGR > {bench_cagr*100:.2f}%, MDD > {bench_mdd*100:.2f}%, Sharpe > {bench_sh:.3f}, Calmar > {bench_cal:.3f}\n")

for name, (c, m, sh, ca, fv, so, eq) in results.items():
    beats_cagr = c > bench_cagr
    beats_mdd = m > bench_mdd
    beats_sh = sh > bench_sh
    beats_cal = ca > bench_cal
    beat_count = sum([beats_cagr, beats_mdd, beats_sh, beats_cal])
    if beat_count == 4:
        flag = "⭐ DOMINATES"
    elif beat_count == 3:
        flag = "✓ 3/4"
    else:
        flag = f"  {beat_count}/4"
    print(f"  {name:<42} {flag} | CAGR {c*100:>6.2f}%  MDD {m*100:>6.2f}%  Sharpe {sh:.3f}  Calmar {ca:.3f}")

# ============================================================
# CPCV on the best candidates
# ============================================================
print(f"\n{'=' * 108}")
print(f"CPCV (N=10, k=2 → 45 paths) — validating top candidates")
print("=" * 108)

def cpcv_run(on_px, off_px, sig_px, n_splits=10, k_test=2, embargo=21):
    n = len(sig_px)
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
            test_idx = sig_px.index[tf_start_emb:tf_end_emb]
            warm_start = max(0, tf_start_emb - SLOW - 50)
            full_idx = sig_px.index[warm_start:tf_end_emb]

            bull = get_bull(sig_px.loc[full_idx], FAST, SLOW).astype(float)
            on_pos = bull.loc[test_idx]
            off_pos = (1 - bull).loc[test_idx]
            on_r = on_px.loc[test_idx].pct_change().fillna(0)
            off_r = off_px.loc[test_idx].pct_change().fillna(0)
            pc = on_pos.diff().abs().fillna(0) + off_pos.diff().abs().fillna(0)
            cost = pc * 7.5 / 10000
            strat_ret = on_pos.shift(1).fillna(0) * on_r + off_pos.shift(1).fillna(0) * off_r - cost
            eq = (1 + strat_ret).cumprod() * INIT
            if not strat_ret.empty:
                test_rets.append(strat_ret)
        if test_rets:
            combined = pd.concat(test_rets)
            combined_eq = (1 + combined).cumprod() * INIT
            c, m, sh, ca, fv, so = metrics(combined_eq)
            path_results.append({'cagr': c, 'mdd': m, 'sharpe': sh, 'calmar': ca})
    return path_results

# Also CPCV QQQ B&H
def cpcv_bh(px, n_splits=10, k_test=2, embargo=21):
    n = len(px)
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
            test_idx = px.index[tf_start_emb:tf_end_emb]
            ret = px.loc[test_idx].pct_change().fillna(0)
            if not ret.empty:
                test_rets.append(ret)
        if test_rets:
            combined = pd.concat(test_rets)
            combined_eq = (1 + combined).cumprod() * INIT
            c, m, sh, ca, fv, so = metrics(combined_eq)
            path_results.append({'cagr': c, 'mdd': m, 'sharpe': sh, 'calmar': ca})
    return path_results

# Run CPCV on all candidates
cpcv_specs = [
    ("QQQ/Cash rotation", qqq_c, pd.Series(1.0, index=qqq_c.index), qqq_c),
    ("QQQ/TLT rotation", qqq_c, tlt_c, qqq_c),
    ("QQQ/AGG rotation", qqq_c, agg_c, qqq_c),
    ("SPY/TLT rotation", spy_c, tlt_c, spy_c),
    ("SPY/AGG rotation", spy_c, agg_c, spy_c),
    ("QQQ→TQQQ/AGG rotation", tqqq_c, agg_c, qqq_c),
    ("QQQ→TQQQ/TLT rotation", tqqq_c, tlt_c, qqq_c),
    ("QQQ→TQQQ/QQQ (baseline)", tqqq_c, qqq_c, qqq_c),
]

cpcv_data = {}
for name, on_px, off_px, sig_px in cpcv_specs:
    r = cpcv_run(on_px, off_px, sig_px)
    cpcv_data[name] = r

# QQQ B&H CPCV
cpcv_data["QQQ Buy & Hold"] = cpcv_bh(qqq_c)

print(f"\n{'Strategy':<32} {'Med CAGR':>10} {'CAGR IQR':>22} {'Med Sharpe':>11} {'Med Calmar':>11} {'Med MDD':>10} {'P(>0)':>7} {'P(>10%)':>9}")
print("-" * 130)
for name in cpcv_data:
    results = cpcv_data[name]
    if not results: continue
    cagrs = np.array([p['cagr'] for p in results])
    sharpes = np.array([p['sharpe'] for p in results])
    calmars = np.array([p['calmar'] for p in results])
    mdds = np.array([p['mdd'] for p in results])
    q25 = np.percentile(cagrs, 25) * 100
    q75 = np.percentile(cagrs, 75) * 100
    iqr = f"[{q25:>6.2f}%, {q75:>6.2f}%]"
    print(f"{name:<32} {np.median(cagrs)*100:>9.2f}% {iqr:>22} {np.median(sharpes):>11.3f} {np.median(calmars):>11.3f} {np.median(mdds)*100:>9.2f}% {(cagrs>0).mean()*100:>6.1f}% {(cagrs>0.10).mean()*100:>8.1f}%")

# Save
import pickle
with open("/tmp/ema_backtest/retail_candidates.pkl", "wb") as f:
    pickle.dump({'single': results, 'cpcv': cpcv_data}, f)
