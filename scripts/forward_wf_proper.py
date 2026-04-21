"""
Causally correct forward walk-forward + Purged/Embargoed CV (Lopez de Prado 2018)

Previous mistake: used 2010-2026 to pick parameters, tested on 1999-2010 -> anti-causal
Correct approach: start from 1999, training uses only historical data, testing only sees the future

Additionally:
- Purged CV: leave embargo period between train/test to prevent label leakage
- Combinatorial Purged Cross-Validation (CPCV)
- Full 1999-2026 single complete forward run
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0

print("[1/6] Loading data ...")
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

def get_bull(close, fast, slow):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s)
    bull.iloc[:slow] = False
    return bull

def build_pyramid_pos(qqq, bull, ma_period, base_tqqq, levels, anchor):
    if anchor == "ma":
        ma = qqq.ewm(span=ma_period, adjust=False).mean()
        deviation = qqq / ma - 1
    else:
        peak = qqq.cummax()
        deviation = qqq / peak - 1
    tpos = pd.Series(0.0, index=qqq.index)
    for i in range(len(qqq.index)):
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

def backtest_dual(qpos, tpos, qqq, tqqq, fee_bps=2.5, slip_bps=5.0):
    qpos_l = qpos.shift(1).fillna(0)
    tpos_l = tpos.shift(1).fillna(0)
    qret = qqq.pct_change().fillna(0)
    tret = tqqq.pct_change().fillna(0)
    pos_change = qpos_l.diff().abs().fillna(0) + tpos_l.diff().abs().fillna(0)
    cost = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = qpos_l * qret + tpos_l * tret - cost
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1 if eq.iloc[-1] > 0 else -1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else cagr
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

# ============================================================
# Causally correct forward walk-forward, starting in 1999
# Each train window uses only past data, test is truly "future"
# ============================================================
def forward_wf(qqq, tqqq, train_y, test_y, embargo_d, strategy_fn, search_grid):
    """
    Forward WF + embargo:
    - 1999 to 1999+train_y: train
    - +embargo_d: isolation period (prevents label leakage)
    - Test test_y years
    - Roll forward
    """
    all_returns = []
    chosen = []
    test_periods = []

    train_d = train_y * 252
    test_d = test_y * 252

    start_idx = train_d
    while start_idx + embargo_d + test_d <= len(qqq):
        train_end = start_idx
        train_start = max(0, train_end - train_d)  # rolling window
        # train_start = 0  # or expanding window

        train_idx = qqq.index[train_start:train_end]
        test_start_idx = train_end + embargo_d  # embargo gap
        test_end_idx = min(test_start_idx + test_d, len(qqq))
        test_idx = qqq.index[test_start_idx:test_end_idx]

        # Train: pick best params on train set
        best_cal = -999; best_p = None
        for p in search_grid:
            try:
                qpos_train, tpos_train = strategy_fn(qqq.loc[train_idx], p)
                eq, ret = backtest_dual(qpos_train, tpos_train, qqq.loc[train_idx], tqqq.loc[train_idx])
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = p
            except: continue

        if best_p is None:
            start_idx += test_d
            continue

        # Test: apply train-chosen params to test period
        # Note: EMA/cummax needs warm-up, so compute from train_start, take only test-period results
        full_idx = qqq.index[train_start:test_end_idx]
        qpos_full, tpos_full = strategy_fn(qqq.loc[full_idx], best_p)
        qpos_test = qpos_full.loc[test_idx]
        tpos_test = tpos_full.loc[test_idx]
        eq, ret = backtest_dual(qpos_test, tpos_test, qqq.loc[test_idx], tqqq.loc[test_idx])
        all_returns.append(ret)
        chosen.append(best_p)
        test_periods.append((test_idx[0], test_idx[-1]))
        start_idx = test_end_idx

    if not all_returns:
        return None, None, [], []
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen, test_periods

# Strategy: WF QQQ/TQQQ rotation
def strat_rotation(qqq_window, p):
    f, s = p
    bull = get_bull(qqq_window, f, s).astype(float)
    qpos = 1 - bull
    tpos = bull
    return qpos, tpos

# Strategy: WF cash/TQQQ
def strat_cash(qqq_window, p):
    f, s = p
    bull = get_bull(qqq_window, f, s).astype(float)
    qpos = pd.Series(0.0, index=qqq_window.index)
    tpos = bull
    return qpos, tpos

# Strategy: Pure TQQQ pyramid (peak-anchored, T4)
def strat_pyramid_t4(qqq_window, p):
    f, s = p
    bull = get_bull(qqq_window, f, s)
    levels = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]
    tpos = build_pyramid_pos(qqq_window, bull, s, 0.0, levels, "peak")
    qpos = pd.Series(0.0, index=qqq_window.index)
    return qpos, tpos

# Strategy: Pure TQQQ pyramid (half-base MA, T6)
def strat_pyramid_t6(qqq_window, p):
    f, s = p
    bull = get_bull(qqq_window, f, s)
    levels = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]
    tpos = build_pyramid_pos(qqq_window, bull, s, 0.5, levels, "ma")
    qpos = pd.Series(0.0, index=qqq_window.index)
    return qpos, tpos

# Search grid
fs_grid = [(f, s) for f in [3, 5, 8, 10, 13] for s in [50, 100, 150, 200, 250] if f < s]

print(f"\n[2/6] Running forward walk-forward (5y train, 2y test, 21d embargo)")
print(f"  Search grid: {len(fs_grid)} (fast, slow) combinations")
print(f"  Strategies: rotation, cash, pyramid-T4, pyramid-T6")
print()

EMBARGO = 21  # 1 month embargo per López de Prado recommendation

strategies_to_test = [
    ("Forward-WF QQQ/TQQQ rotation", strat_rotation),
    ("Forward-WF TQQQ/cash", strat_cash),
    ("Forward-WF Pure TQQQ pyramid (T4)", strat_pyramid_t4),
    ("Forward-WF Pure TQQQ pyramid (T6)", strat_pyramid_t6),
]

results = {}
for name, fn in strategies_to_test:
    print(f"  Running {name} ...")
    eq, ret, chosen, periods = forward_wf(qqq, tqqq_full, 5, 2, EMBARGO, fn, fs_grid)
    if eq is None:
        print(f"    FAILED")
        continue
    c, m, sh, ca, fv, so = metrics(eq, ret)
    results[name] = {
        'eq': eq, 'ret': ret, 'chosen': chosen, 'periods': periods,
        'metrics': (c, m, sh, ca, fv, so)
    }
    print(f"    CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Calmar {ca:.3f}")

# ============================================================
# Embargoed forward CV with different train/test splits
# ============================================================
print(f"\n[3/6] Embargoed forward CV: testing different (train, test) windows")
print(f"  All forward, no future leakage, 21-day embargo")
print()

fwd_cv_results = []
for tr_y in [3, 5, 7]:
    for te_y in [1, 2, 3]:
        eq, ret, _, _ = forward_wf(qqq, tqqq_full, tr_y, te_y, EMBARGO, strat_rotation, fs_grid)
        if eq is None: continue
        c, m, sh, ca, fv, so = metrics(eq, ret)
        fwd_cv_results.append((tr_y, te_y, c, m, sh, ca, fv))

print(f"  {'Train':<8} {'Test':<8} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'Final $10K':>14}")
print("  " + "-" * 76)
for tr, te, c, m, sh, ca, fv in fwd_cv_results:
    print(f"  {tr}y      {te}y     {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {fv:>14,.0f}")

print()
print(f"  Median across all (train, test) configs:")
cagrs = [r[2] for r in fwd_cv_results]
calmars = [r[5] for r in fwd_cv_results]
print(f"    CAGR median: {np.median(cagrs)*100:.2f}%, IQR [{np.percentile(cagrs, 25)*100:.2f}%, {np.percentile(cagrs, 75)*100:.2f}%]")
print(f"    Calmar median: {np.median(calmars):.3f}, IQR [{np.percentile(calmars, 25):.3f}, {np.percentile(calmars, 75):.3f}]")

# ============================================================
# Combinatorial Purged Cross-Validation (CPCV)
# López de Prado 2018, Advances in Financial ML, Ch 7
# ============================================================
print(f"\n[4/6] Combinatorial Purged Cross-Validation (CPCV)")
print(f"  N=10 splits, k=2 test groups -> C(10,2) = 45 backtest paths")
print(f"  Per López de Prado 2018, addresses overlap between train/test in serial data")
print()

def cpcv(qqq, tqqq, n_splits, k_test, embargo_d, strategy_fn, search_grid):
    """Combinatorial Purged Cross-Validation"""
    n = len(qqq)
    fold_size = n // n_splits
    folds = [(i * fold_size, (i+1) * fold_size if i < n_splits-1 else n) for i in range(n_splits)]

    all_combos = list(combinations(range(n_splits), k_test))
    path_results = []

    for combo in all_combos:
        train_folds = [i for i in range(n_splits) if i not in combo]
        test_folds = list(combo)

        # Train indices
        train_idx_list = []
        for i in train_folds:
            train_idx_list.extend(range(folds[i][0], folds[i][1]))
        train_idx = qqq.index[train_idx_list]

        if len(train_idx) < 252:  # Need at least 1 year
            continue

        # Train: select best params
        best_cal = -999; best_p = None
        for p in search_grid:
            try:
                qpos_t, tpos_t = strategy_fn(qqq.loc[train_idx], p)
                eq, ret = backtest_dual(qpos_t, tpos_t, qqq.loc[train_idx], tqqq.loc[train_idx])
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = p
            except: continue

        if best_p is None: continue

        # Test on test folds (with purging + embargo)
        test_returns = []
        for tf in test_folds:
            tf_start, tf_end = folds[tf]
            # Apply embargo: shrink test by embargo_d on each side
            tf_start_emb = tf_start + embargo_d
            tf_end_emb = max(tf_start_emb + 1, tf_end - embargo_d)
            if tf_end_emb - tf_start_emb < 100: continue

            test_idx = qqq.index[tf_start_emb:tf_end_emb]
            # Need warm-up from before
            warm_start = max(0, tf_start_emb - 252)
            full_idx = qqq.index[warm_start:tf_end_emb]
            qpos_f, tpos_f = strategy_fn(qqq.loc[full_idx], best_p)
            qpos_test = qpos_f.loc[test_idx]
            tpos_test = tpos_f.loc[test_idx]
            eq, ret = backtest_dual(qpos_test, tpos_test, qqq.loc[test_idx], tqqq.loc[test_idx])
            if not ret.empty:
                test_returns.append(ret)

        if test_returns:
            combined = pd.concat(test_returns)
            combined_eq = (1 + combined).cumprod() * INIT_CASH
            c, m, sh, ca, fv, so = metrics(combined_eq, combined)
            path_results.append({'combo': combo, 'best_p': best_p,
                                  'cagr': c, 'mdd': m, 'sharpe': sh, 'calmar': ca})

    return path_results

print(f"  Running CPCV on rotation strategy ...")
cpcv_results = cpcv(qqq, tqqq_full, n_splits=10, k_test=2, embargo_d=EMBARGO,
                     strategy_fn=strat_rotation, search_grid=fs_grid)
print(f"  Got {len(cpcv_results)} valid backtest paths")

cagrs = [r['cagr'] for r in cpcv_results]
sharpes = [r['sharpe'] for r in cpcv_results]
calmars = [r['calmar'] for r in cpcv_results]
mdds = [r['mdd'] for r in cpcv_results]

print(f"\n  Distribution across {len(cpcv_results)} CPCV paths (rotation strategy):")
print(f"    CAGR:   median {np.median(cagrs)*100:>6.2f}%, IQR [{np.percentile(cagrs, 25)*100:>6.2f}%, {np.percentile(cagrs, 75)*100:>6.2f}%]")
print(f"    Sharpe: median {np.median(sharpes):>6.3f}, IQR [{np.percentile(sharpes, 25):>6.3f}, {np.percentile(sharpes, 75):>6.3f}]")
print(f"    Calmar: median {np.median(calmars):>6.3f}, IQR [{np.percentile(calmars, 25):>6.3f}, {np.percentile(calmars, 75):>6.3f}]")
print(f"    MDD:    median {np.median(mdds)*100:>6.2f}%, IQR [{np.percentile(mdds, 25)*100:>6.2f}%, {np.percentile(mdds, 75)*100:>6.2f}%]")
print(f"    P(CAGR > 15%): {np.mean(np.array(cagrs) > 0.15)*100:.1f}%")
print(f"    P(Sharpe > 0.5): {np.mean(np.array(sharpes) > 0.5)*100:.1f}%")

# ============================================================
# CPCV for pyramid strategies (the ones that "failed" in anti-causal test)
# ============================================================
print(f"\n[5/6] CPCV on pyramid strategies (the ones I previously claimed failed)")
print()

for strat_name, fn in [("Pyramid T4 (peak-anchored)", strat_pyramid_t4),
                          ("Pyramid T6 (half-base MA)", strat_pyramid_t6),
                          ("Cash/TQQQ", strat_cash)]:
    cpcv_r = cpcv(qqq, tqqq_full, 10, 2, EMBARGO, fn, fs_grid)
    if not cpcv_r:
        print(f"  {strat_name}: NO VALID PATHS")
        continue
    cagrs = np.array([r['cagr'] for r in cpcv_r])
    calmars = np.array([r['calmar'] for r in cpcv_r])
    mdds = np.array([r['mdd'] for r in cpcv_r])
    print(f"  {strat_name} ({len(cpcv_r)} paths):")
    print(f"    CAGR:   median {np.median(cagrs)*100:>6.2f}%, IQR [{np.percentile(cagrs, 25)*100:>6.2f}%, {np.percentile(cagrs, 75)*100:>6.2f}%]")
    print(f"    Calmar: median {np.median(calmars):>6.3f}, IQR [{np.percentile(calmars, 25):>6.3f}, {np.percentile(calmars, 75):>6.3f}]")
    print(f"    MDD:    median {np.median(mdds)*100:>6.2f}%, worst {np.min(mdds)*100:>6.2f}%")
    print(f"    P(CAGR > 0): {np.mean(cagrs > 0)*100:.1f}%, P(CAGR > 15%): {np.mean(cagrs > 0.15)*100:.1f}%")

# ============================================================
# Final comparison: all strategies with proper forward methodology
# ============================================================
print(f"\n[6/6] FINAL COMPARISON (all causally clean):")
print()

print(f"  {'Strategy':<48} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'$10K Final':>14}")
print("  " + "-" * 105)

# Forward WF results
for name in ["Forward-WF QQQ/TQQQ rotation",
              "Forward-WF TQQQ/cash",
              "Forward-WF Pure TQQQ pyramid (T4)",
              "Forward-WF Pure TQQQ pyramid (T6)"]:
    if name in results:
        c, m, sh, ca, fv, so = results[name]['metrics']
        print(f"  {name:<48} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {fv:>14,.0f}")

# Buy & hold benchmarks (full 1999-2026)
for name, px in [("TQQQ Buy & Hold (1999-2026, synthetic+real)", tqqq_full),
                  ("QQQ Buy & Hold (1999-2026)", qqq)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"  {name:<48} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {fv:>14,.0f}")

print()
print("=" * 100)
print("KEY CHANGE FROM PREVIOUS ANALYSIS:")
print("  - Old (WRONG): trained on 2010-2026, tested on 1999-2010 (anti-causal, future-info leakage)")
print("  - New (RIGHT): forward walk-forward starts from 1999, uses ONLY past data at every test")
print("  - Pyramid strategies get to 'learn' from dot-com period in their training set")
print("  - This is the modern best practice (López de Prado 2018, Bailey & López de Prado 2014)")

# Save results to JSON for chart generation
import json
out = {}
for name, r in results.items():
    c, m, sh, ca, fv, so = r['metrics']
    out[name] = {'CAGR': float(c), 'MDD': float(m), 'Sharpe': float(sh),
                 'Sortino': float(so), 'Calmar': float(ca), 'Final': float(fv)}
    # Save equity curve as date-indexed series
    out[name]['equity_dates'] = [d.strftime("%Y-%m-%d") for d in r['eq'].index]
    out[name]['equity_values'] = r['eq'].values.tolist()

import pickle
with open("/tmp/ema_backtest/forward_wf_results.pkl", "wb") as f:
    pickle.dump({'results': results, 'cpcv_rotation': cpcv_results,
                 'fwd_cv': fwd_cv_results}, f)
print(f"\nResults saved to /tmp/ema_backtest/forward_wf_results.pkl")
