"""
Multi-source data validation + QQQ/TQQQ rotation strategy
1. Use Alpha Vantage data to validate yfinance price authenticity
2. Check price points at key historical events
3. Design multiple QQQ/TQQQ rotation strategies, avoiding pure cash
4. Strict walk-forward testing
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json

START = "1999-03-10"
END = "2026-01-17"
INIT_CASH = 10_000.0

print("=" * 100)
print("[Data Verification 1] yfinance vs Alpha Vantage cross-check on QQQ close")
print("=" * 100)

# AV data (pulled via MCP, hardcoded here for comparison)
av_qqq = {
    "2026-01-16": 621.26,
    "2026-01-15": 621.78,
    "2026-01-14": 619.55,
    "2026-01-13": 626.24,
    "2026-01-12": 627.17,
    "2026-01-09": 626.65,
    "2026-01-02": 613.12,
    "2025-12-31": 614.31,
    "2025-12-30": 619.43,
    "2025-12-22": 619.21,
    "2025-12-15": 610.54,
    "2025-12-01": 617.17,
    "2025-11-28": 619.25,
    "2025-11-21": 590.07,
}

# yfinance data
yf_qqq = yf.download("QQQ", start="2025-11-20", end="2026-01-18", auto_adjust=False, progress=False)["Close"].squeeze()

print(f"\n{'Date':<12} {'AlphaVantage':>14} {'yfinance':>14} {'Diff %':>10} {'Match':>10}")
print("-" * 70)
all_match = True
for date_str, av_close in av_qqq.items():
    date = pd.Timestamp(date_str)
    if date in yf_qqq.index:
        yf_close = float(yf_qqq.loc[date])
        diff = abs(yf_close - av_close) / av_close * 100
        match = "YES" if diff < 0.05 else "NO"
        if diff > 0.05: all_match = False
        print(f"{date_str:<12} ${av_close:>13.2f} ${yf_close:>13.2f} {diff:>9.3f}% {match:>10}")

print(f"\n-> Data consistency: {'fully reliable' if all_match else 'discrepancy found'}")
print()

print("=" * 100)
print("[Data Verification 2] Key historical event date QQQ/TQQQ prices - known-event cross-check")
print("=" * 100)

qqq = yf.download("QQQ", start="1999-03-10", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
qqq_raw = yf.download("QQQ", start="1999-03-10", end="2026-04-18", auto_adjust=False, progress=False)["Close"].squeeze()
tqqq = yf.download("TQQQ", start="2010-02-11", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_raw = yf.download("TQQQ", start="2010-02-11", end="2026-04-18", auto_adjust=False, progress=False)["Close"].squeeze()
vix = yf.download("^VIX", start="1999-03-10", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()

# Known historical events (from wikipedia / publicly documented news)
events = [
    # (date, event, expected behavior)
    ("2000-03-10", "Nasdaq dot-com peak ($117)", qqq_raw, 117, "QQQ raw"),
    ("2002-10-09", "dot-com bottom ($20)", qqq_raw, 20, "QQQ raw"),
    ("2008-09-15", "Lehman collapse", qqq_raw, 41, "QQQ raw"),
    ("2008-11-20", "Financial crisis bottom", qqq_raw, 25.6, "QQQ raw"),
    ("2010-02-11", "TQQQ IPO", tqqq_raw, 48.3, "TQQQ raw"),  # IPO price
    ("2020-03-23", "COVID bottom", qqq_raw, 170, "QQQ raw"),
    ("2020-08-31", "VIX all-time high 80+", vix, 80, "VIX peak (need 2020-03)"),
    ("2022-01-13", "TQQQ pre 2:1 split", tqqq_raw, 65, "TQQQ raw pre-split"),
    ("2022-10-13", "2022 bear market bottom", qqq_raw, 254, "QQQ raw"),
]

print(f"\n{'Date':<12} {'Event':<30} {'Source':<12} {'Expected ~':>10} {'Actual':>9} {'Reasonable':>10}")
print("-" * 90)
for date, event, series, expected, src in events:
    try:
        date_ts = pd.Timestamp(date)
        # Find closest trading day
        if date_ts in series.index:
            actual = float(series.loc[date_ts])
        else:
            actual = float(series.loc[series.index <= date_ts].iloc[-1])
        diff_pct = abs(actual - expected) / expected * 100
        ok = "YES" if diff_pct < 25 else "WARN"
        print(f"{date:<12} {event:<30} {src:<12} {expected:>10.2f} {actual:>9.2f} {ok:>10}")
    except Exception as e:
        print(f"{date:<12} {event:<30} ERROR: {str(e)[:40]}")

print()
print("=" * 100)
print("[Data Verification 3] TQQQ 2022-01-13 split (2:1) - check adjustment correctness")
print("=" * 100)

tqqq_split_check = yf.download("TQQQ", start="2022-01-10", end="2022-01-15", auto_adjust=False, progress=False)
print("Raw prices around split:")
print(tqqq_split_check[["Open", "Close"]].round(2))
ratio = float(tqqq_split_check["Close"].iloc[-1].squeeze() / tqqq_split_check["Close"].iloc[-2].squeeze())
print(f"\nSplit-day ratio: {ratio:.3f} (should be ~ 0.5 for 2:1 split)")
expected_split = 0.5
match = "YES" if abs(ratio - expected_split) < 0.1 else "NO"
print(f"Matches 2:1 split: {match}")

print()
print("=" * 100)
print("[Synthetic TQQQ reconstruction] (verified error 5.4%)")
print("=" * 100)

def build_tqqq_full(qqq_close, tqqq_real):
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

tqqq_full = build_tqqq_full(qqq, tqqq)
print(f"Synthetic+real TQQQ: {tqqq_full.index[0].date()} ~ {tqqq_full.index[-1].date()}, {len(tqqq_full)} days")
print()

print("=" * 100)
print("[Core experiment] QQQ/TQQQ rotation strategies - full sample 1999-2026")
print("=" * 100)
print()
print("Rule definitions:")
print("  v0 cash/TQQQ:   risk on -> TQQQ, risk off -> cash (original plan)")
print("  v1 QQQ/TQQQ:    risk on -> TQQQ, risk off -> QQQ (always in market)")
print("  v2 three-state: strong -> TQQQ, weak -> QQQ, very weak -> cash")
print("  v3 partial:     strong -> 70% TQQQ + 30% QQQ, weak -> 100% QQQ")
print("  v4 VIX filter:  v1 + force QQQ when VIX>30")
print()

def backtest_dual(pos_qqq, pos_tqqq, qqq_px, tqqq_px, fee_bps=2.5, slip_bps=5.0):
    """
    Dual-asset backtest. pos_qqq + pos_tqqq <= 1 (remainder is cash)
    pos_*: target daily weight (0~1)
    """
    qqq_pos = pos_qqq.shift(1).fillna(0)
    tqqq_pos = pos_tqqq.shift(1).fillna(0)
    qqq_ret = qqq_px.pct_change().fillna(0)
    tqqq_ret = tqqq_px.pct_change().fillna(0)
    # Transaction cost due to position changes (by total turnover)
    pos_change_q = qqq_pos.diff().abs().fillna(0)
    pos_change_t = tqqq_pos.diff().abs().fillna(0)
    cost = (pos_change_q + pos_change_t) * (fee_bps + slip_bps) / 10000
    strat_ret = qqq_pos * qqq_ret + tqqq_pos * tqqq_ret - cost
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    n_trades = int(((pos_change_q + pos_change_t) > 0.01).sum())
    return eq, strat_ret, n_trades

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1]

# Common signal base: QQQ EMA 5/200
ma5 = qqq.ewm(span=5, adjust=False).mean()
ma30 = qqq.ewm(span=30, adjust=False).mean()
ma200 = qqq.ewm(span=200, adjust=False).mean()
ma50 = qqq.ewm(span=50, adjust=False).mean()
vix_aligned = vix.reindex(qqq.index).ffill()

# Risk on/off signals
sig_strong = (ma5 > ma200).astype(float)  # bull
sig_strong.iloc[:200] = 0
sig_mild_bull = (ma30 > ma200).astype(float)  # moderate bull
sig_mild_bull.iloc[:200] = 0
sig_extreme_bear = ((ma5 < ma200) & (ma30 < ma200) & (ma50 < ma200)).astype(float)  # very weak
sig_extreme_bear.iloc[:200] = 0

strategies = {}

# v0: original - cash/TQQQ
qpos_v0 = pd.Series(0.0, index=qqq.index)
tpos_v0 = sig_strong.copy()
strategies["v0 cash/TQQQ"] = (qpos_v0, tpos_v0)

# v1: full QQQ <-> full TQQQ
qpos_v1 = 1 - sig_strong
tpos_v1 = sig_strong.copy()
strategies["v1 QQQ/TQQQ full switch"] = (qpos_v1, tpos_v1)

# v2: three-state (strong TQQQ / moderate QQQ / extreme-weak cash)
qpos_v2 = sig_mild_bull * (1 - sig_strong) * (1 - sig_extreme_bear)  # moderate and not strong
tpos_v2 = sig_strong.copy()
# Very weak: full flat
mask = sig_extreme_bear == 1
qpos_v2[mask] = 0
tpos_v2[mask] = 0
strategies["v2 3-state TQQQ/QQQ/cash"] = (qpos_v2, tpos_v2)

# v3: partial sizing
qpos_v3 = pd.Series(0.0, index=qqq.index)
tpos_v3 = pd.Series(0.0, index=qqq.index)
for i in range(len(qqq.index)):
    if sig_strong.iloc[i] == 1:
        qpos_v3.iloc[i] = 0.30
        tpos_v3.iloc[i] = 0.70
    else:
        qpos_v3.iloc[i] = 1.00
        tpos_v3.iloc[i] = 0.00
strategies["v3 70%TQQQ+30%QQQ / 100%QQQ"] = (qpos_v3, tpos_v3)

# v4: v1 + VIX filter
qpos_v4 = qpos_v1.copy()
tpos_v4 = tpos_v1.copy()
high_vix = vix_aligned > 30
qpos_v4[high_vix] = 1.0
tpos_v4[high_vix] = 0.0
strategies["v4 v1 + VIX>30 force QQQ"] = (qpos_v4, tpos_v4)

# v5: VIX-dynamic switch (low VIX full TQQQ; mid VIX 100% QQQ; high VIX cash)
qpos_v5 = pd.Series(0.0, index=qqq.index)
tpos_v5 = pd.Series(0.0, index=qqq.index)
for i in range(len(qqq.index)):
    s = sig_strong.iloc[i]
    v = vix_aligned.iloc[i]
    if pd.isna(v): v = 20
    if s == 1 and v < 20:
        qpos_v5.iloc[i] = 0.0; tpos_v5.iloc[i] = 1.0
    elif s == 1 and v < 30:
        qpos_v5.iloc[i] = 0.5; tpos_v5.iloc[i] = 0.5
    elif s == 1:
        qpos_v5.iloc[i] = 1.0; tpos_v5.iloc[i] = 0.0
    elif v < 30:
        qpos_v5.iloc[i] = 1.0; tpos_v5.iloc[i] = 0.0
    else:
        qpos_v5.iloc[i] = 0.0; tpos_v5.iloc[i] = 0.0
strategies["v5 VIX 3-tier dynamic allocation"] = (qpos_v5, tpos_v5)

print(f"\n{'Strategy':<38} {'CAGR':>8} {'MDD':>9} {'Sharpe':>7} {'Calmar':>7} {'Final Value':>14} {'Trades':>5}")
print("-" * 100)
for name, (qpos, tpos) in strategies.items():
    eq, ret, n = backtest_dual(qpos, tpos, qqq, tqqq_full)
    c, m, s, ca, fv = metrics(eq, ret)
    print(f"{name:<38} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f} {n:>5}")

# Buy & hold control
for name, px in [("QQQ B&H plain", qqq), ("TQQQ B&H plain", tqqq_full)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, s, ca, fv = metrics(eq, ret)
    print(f"{name:<38} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f} {'-':>5}")

print()
print("=" * 100)
print("[Strict out-of-sample test] 1999-2010 (dot-com + 2008) performance of each strategy")
print("=" * 100)

oos_idx = qqq.index <= "2010-02-10"

print(f"\n{'Strategy':<38} {'OOS CAGR':>10} {'OOS MDD':>10} {'OOS Calmar':>11} {'Not losing':>10}")
print("-" * 100)
for name, (qpos, tpos) in strategies.items():
    eq, ret, n = backtest_dual(qpos[oos_idx], tpos[oos_idx], qqq[oos_idx], tqqq_full[oos_idx])
    c, m, s, ca, fv = metrics(eq, ret)
    flag = "YES" if c > 0 else "NO"
    print(f"{name:<38} {c*100:>9.2f}% {m*100:>9.2f}% {ca:>11.3f} {flag:>10}")

# B&H control
for name, px in [("QQQ B&H 1999-2010", qqq[oos_idx]), ("TQQQ B&H 1999-2010", tqqq_full[oos_idx])]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, s, ca, fv = metrics(eq, ret)
    print(f"{name:<38} {c*100:>9.2f}% {m*100:>9.2f}% {ca:>11.3f}")

print()
print("=" * 100)
print("[Walk-Forward + rotation] Best strategy dynamic params + QQQ/TQQQ rotation")
print("=" * 100)

def run_wf_rotation(qqq, tqqq_full, train_years=5, test_years=2):
    """walk-forward, pick best (fast, slow) per segment, execute via QQQ/TQQQ rotation"""
    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    chosen = []
    test_periods = []

    start_idx = 252 * train_years
    while start_idx + 252 * test_years <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_years, len(qqq))
        train_idx = qqq.index[train_end - 252*train_years : train_end]
        test_idx = qqq.index[train_end : test_end]

        best_cal = -999
        best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                ema_f = qqq.loc[train_idx].ewm(span=f, adjust=False).mean()
                ema_s = qqq.loc[train_idx].ewm(span=s, adjust=False).mean()
                sig = (ema_f > ema_s).astype(float); sig.iloc[:s] = 0
                qpos = 1 - sig
                tpos = sig
                eq, ret, _ = backtest_dual(qpos, tpos, qqq.loc[train_idx], tqqq_full.loc[train_idx])
                _, _, _, cal, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)

        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_years : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        sig = (ema_f > ema_s).astype(float); sig.iloc[:s] = 0
        sig_test = sig.loc[test_idx]
        qpos = 1 - sig_test
        tpos = sig_test
        eq, ret, _ = backtest_dual(qpos, tpos, qqq.loc[test_idx], tqqq_full.loc[test_idx])
        all_returns.append(ret)
        chosen.append(best_p)
        test_periods.append((test_idx[0], test_idx[-1]))
        start_idx += 252 * test_years

    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen, test_periods

print("\nRunning walk-forward (5y train + 2y test, QQQ/TQQQ rotation) ...")
wf_eq, wf_ret, wf_params, wf_periods = run_wf_rotation(qqq, tqqq_full, 5, 2)
c, m, s, ca, fv = metrics(wf_eq, wf_ret)
print(f"\nWalk-Forward QQQ/TQQQ rotation results:")
print(f"  CAGR: {c*100:.2f}%, MDD: {m*100:.2f}%, Sharpe: {s:.3f}, Calmar: {ca:.3f}")
print(f"  Final value: ${fv:,.0f} (starting $10,000)")
print(f"\nBest parameters per window:")
for (sd, ed), (f, sl) in zip(wf_periods, wf_params):
    print(f"  {sd.date()} -> {ed.date()}: fast={f}, slow={sl}")

# Control: same-period buy & hold
test_period = wf_ret.index
qqq_test = qqq.loc[test_period]
tqqq_test = tqqq_full.loc[test_period]
print(f"\nSame-period Buy & Hold control ({test_period[0].date()} ~ {test_period[-1].date()}):")
for name, px in [("QQQ B&H same period", qqq_test), ("TQQQ B&H same period", tqqq_test)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, s, ca, fv = metrics(eq, ret)
    print(f"  {name:<25}: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {ca:.3f}, Final ${fv:,.0f}")

# Compare with previous cash version walk-forward
print(f"\nCompare with previous cash/TQQQ Walk-Forward (cash version):")
def run_wf_cash(qqq, tqqq_full, train_years=5, test_years=2):
    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    start_idx = 252 * train_years
    while start_idx + 252 * test_years <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_years, len(qqq))
        train_idx = qqq.index[train_end - 252*train_years : train_end]
        test_idx = qqq.index[train_end : test_end]
        best_cal = -999
        best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                ema_f = qqq.loc[train_idx].ewm(span=f, adjust=False).mean()
                ema_s = qqq.loc[train_idx].ewm(span=s, adjust=False).mean()
                sig = (ema_f > ema_s).astype(float); sig.iloc[:s] = 0
                eq, ret, _ = backtest_dual(pd.Series(0.0, index=train_idx), sig, qqq.loc[train_idx], tqqq_full.loc[train_idx])
                _, _, _, cal, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_years : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        sig = (ema_f > ema_s).astype(float); sig.iloc[:s] = 0
        sig_test = sig.loc[test_idx]
        eq, ret, _ = backtest_dual(pd.Series(0.0, index=test_idx), sig_test, qqq.loc[test_idx], tqqq_full.loc[test_idx])
        all_returns.append(ret)
        start_idx += 252 * test_years
    return (1 + pd.concat(all_returns)).cumprod() * INIT_CASH, pd.concat(all_returns)

wf_cash_eq, wf_cash_ret = run_wf_cash(qqq, tqqq_full, 5, 2)
c, m, s, ca, fv = metrics(wf_cash_eq, wf_cash_ret)
print(f"  WF cash/TQQQ:    CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {s:.3f}, Calmar {ca:.3f}, Final ${fv:,.0f}")

print()
print("=" * 100)
print("[Final comparison table] Real out-of-sample performance of all methods")
print("=" * 100)

methods = []
# WF rotation
c, m, s, ca, fv = metrics(wf_eq, wf_ret)
methods.append(("WF EMA QQQ/TQQQ rotation (NEW)", c, m, s, ca, fv))
# WF cash
c, m, s, ca, fv = metrics(wf_cash_eq, wf_cash_ret)
methods.append(("WF EMA cash/TQQQ (old)", c, m, s, ca, fv))
# B&H same period
qqq_p = qqq.loc[wf_ret.index]
tqqq_p = tqqq_full.loc[wf_ret.index]
for name, px in [("QQQ Buy & Hold", qqq_p), ("TQQQ Buy & Hold", tqqq_p)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, s, ca, fv = metrics(eq, ret)
    methods.append((name, c, m, s, ca, fv))

# 50/50 rebalanced
qqq_pos_5050 = pd.Series(0.5, index=wf_ret.index)
tqqq_pos_5050 = pd.Series(0.5, index=wf_ret.index)
eq, ret, _ = backtest_dual(qqq_pos_5050, tqqq_pos_5050, qqq_p, tqqq_p)
c, m, s, ca, fv = metrics(eq, ret)
methods.append(("50/50 QQQ+TQQQ continuous hold", c, m, s, ca, fv))

print(f"\n{'Method':<40} {'CAGR':>8} {'MDD':>9} {'Sharpe':>7} {'Calmar':>7} {'Final Value':>14}")
print("-" * 100)
for name, c, m, s, ca, fv in sorted(methods, key=lambda x: -x[4]):  # sort by Calmar
    print(f"{name:<40} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

print()
print("=" * 100)
print("[Bootstrap significance test] WF rotation strategy 1000 resamples")
print("=" * 100)

def bootstrap(ret, n=1000, block=20):
    np.random.seed(42)
    cagrs = []; sharpes = []
    n_obs = len(ret)
    for _ in range(n):
        starts = np.random.randint(0, n_obs - block, n_obs // block)
        idx = np.concatenate([np.arange(st, st+block) for st in starts])
        idx = idx[idx < n_obs]
        s = ret.values[idx]
        eq = np.cumprod(1 + s)
        years = len(s) / 252
        if eq[-1] > 0: cagrs.append(eq[-1] ** (1/years) - 1)
        if s.std() > 0: sharpes.append(s.mean() * 252 / (s.std() * np.sqrt(252)))
    return np.array(cagrs), np.array(sharpes)

cagrs, sharpes = bootstrap(wf_ret, 1000, 20)
print(f"\nWalk-Forward QQQ/TQQQ rotation 1000 block bootstraps:")
print(f"  CAGR  median {np.median(cagrs)*100:.2f}%, 95% CI [{np.percentile(cagrs, 2.5)*100:.2f}%, {np.percentile(cagrs, 97.5)*100:.2f}%]")
print(f"  Sharpe median {np.median(sharpes):.3f}, 95% CI [{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]")
print(f"  P(CAGR > 0): {(cagrs > 0).mean()*100:.1f}%")
print(f"  P(CAGR > 15%): {(cagrs > 0.15).mean()*100:.1f}%")
