"""
Synthetic TQQQ v2 - corrected cost model + strict anti-overfitting analysis
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "1999-03-10"
END = "2026-01-17"
INIT_CASH = 10_000.0

print("[1/6] Downloading full-history QQQ + real TQQQ ...")
qqq_full = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()

# Corrected synthetic model
def daily_financing(date):
    """Fed funds rate annualized -> daily"""
    y = date.year
    if y <= 2007: rate = 0.045
    elif y <= 2008: rate = 0.025
    elif y <= 2015: rate = 0.0015
    elif y <= 2019: rate = 0.015
    elif y <= 2021: rate = 0.001
    else: rate = 0.045
    # Borrow 2x principal to achieve 3x leverage, financing = rate * 2
    return (rate + 0.004) * 2 / 252  # +40bp spread

print("[2/6] Building synthetic TQQQ (corrected costs)")
qqq_ret = qqq_full.pct_change().fillna(0)
expense_daily = 0.0084 / 252  # 0.84%/year
financing_daily = pd.Series([daily_financing(d) for d in qqq_full.index], index=qqq_full.index)
slip_annual = 0.003  # 0.3%/year rebalancing slippage (conservative estimate)
slip_daily = slip_annual / 252

tqqq_synth_ret = 3 * qqq_ret - expense_daily - financing_daily - slip_daily
tqqq_synth = (1 + tqqq_synth_ret).cumprod()

# Validate synthetic vs real
overlap_start = tqqq_real.index[0]
synth_overlap = tqqq_synth.loc[overlap_start:]
real_overlap = tqqq_real.loc[overlap_start:]
synth_norm = synth_overlap / synth_overlap.iloc[0]
real_norm = real_overlap / real_overlap.iloc[0]
final_synth = float(synth_norm.iloc[-1])
final_real = float(real_norm.iloc[-1])
err = abs(final_synth - final_real) / final_real * 100
print(f"  Synthetic TQQQ 16-year: {final_synth:.1f}x")
print(f"  Real TQQQ 16-year: {final_real:.1f}x")
print(f"  Error: {err:.1f}%")

# If there is still error, apply multiplicative calibration: make 16-year final match
calibration = final_real / final_synth
implied_extra_drag = (calibration ** (-1/16) - 1) * 100
print(f"  Implied extra annualized drift: {implied_extra_drag:+.2f}%/year")
# Apply calibration
tqqq_synth_calibrated = tqqq_synth * (calibration ** (np.arange(len(tqqq_synth)) / len(tqqq_synth)))
synth_cal_final = float(tqqq_synth_calibrated.loc[overlap_start:].iloc[-1] / tqqq_synth_calibrated.loc[overlap_start])
print(f"  Calibrated synthetic 16-year: {synth_cal_final:.1f}x (target {final_real:.1f}x)")

# Splice: use calibrated synthetic for 1999-2010, real from 2010+
scale = float(real_overlap.iloc[0]) / float(tqqq_synth_calibrated.loc[overlap_start])
tqqq_pre2010 = tqqq_synth_calibrated.loc[:overlap_start] * scale
tqqq_combined = pd.concat([tqqq_pre2010.iloc[:-1], tqqq_real])
print(f"  Spliced TQQQ: {tqqq_combined.index[0].date()} ~ {tqqq_combined.index[-1].date()}, {len(tqqq_combined)} days")
print()

def backtest_clean(close_sig, close_exec, fast, slow, fee_bps=2.5, slip_bps=5.0):
    ema_f = close_sig.ewm(span=fast, adjust=False).mean()
    ema_s = close_sig.ewm(span=slow, adjust=False).mean()
    sig = (ema_f > ema_s).astype(int)
    sig.iloc[:slow] = 0
    pos = sig.shift(1).fillna(0)
    daily_ret = close_exec.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    n_trades = int((pos.diff().abs() > 0).sum())
    return eq, strat_ret, n_trades

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1]

print("[3/6] Three-segment sample test: in-sample / out-of-sample / full sample")
print("=" * 100)

periods = {
    "Out-of-sample: 1999-2010 (dot-com -82%, 2008 -56%)": tqqq_combined.loc[:"2010-02-10"],
    "In-sample: 2010-2026 (big bull market)": tqqq_real,
    "Full sample: 1999-2026 (across multiple cycles)": tqqq_combined,
}

print(f"\n{'Period':<48} {'Strat CAGR':>10} {'Strat MDD':>10} {'Calmar':>7} {'B&H CAGR':>10} {'B&H MDD':>10}")
print("-" * 110)
for period_name, px in periods.items():
    eq, ret, n = backtest_clean(px, px, 5, 200)
    s_cagr, s_mdd, _, s_cal, _ = metrics(eq, ret)
    bh_eq = (px / px.iloc[0]) * INIT_CASH
    bh_ret = bh_eq.pct_change().fillna(0)
    b_cagr, b_mdd, _, _, _ = metrics(bh_eq, bh_ret)
    print(f"{period_name:<48} {s_cagr*100:>9.2f}% {s_mdd*100:>9.2f}% {s_cal:>7.3f} {b_cagr*100:>9.2f}% {b_mdd*100:>9.2f}%")

print()
print("[4/6] Parameter stability heatmap (key anti-overfitting check)")
print("=" * 100)
print("Rule: ideal parameters sit on a 'plateau'; neighbors (f +/-2, s +/-50) should also be close")
print("If a point's Calmar is far higher than neighbors -> overfitting warning")
print()

fast_grid = [3, 5, 8, 10, 13, 20, 30]
slow_grid = [50, 100, 150, 200, 250, 300]

# Use full sample (with both crashes) for stability
test_px = tqqq_combined

print("--- Full sample 1999-2026 Calmar ratio ---")
print(f"{'fast\\slow':<10}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
calmar_grid = {}
for f in fast_grid:
    print(f"{f:<10}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        eq, ret, n = backtest_clean(test_px, test_px, f, s)
        _, _, _, cal, _ = metrics(eq, ret)
        calmar_grid[(f, s)] = cal
        print(f"{cal:>9.3f}", end="")
    print()

print()
print("--- Out-of-sample 1999-2010 only Calmar (true overfitting detector) ---")
test_px2 = tqqq_combined.loc[:"2010-02-10"]
print(f"{'fast\\slow':<10}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
calmar_oos = {}
for f in fast_grid:
    print(f"{f:<10}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        eq, ret, n = backtest_clean(test_px2, test_px2, f, s)
        _, _, _, cal, _ = metrics(eq, ret)
        calmar_oos[(f, s)] = cal
        print(f"{cal:>9.3f}", end="")
    print()

print()
print("[5/6] Cross-asset robustness (can the same parameters win on SPY/UPRO?)")
print("=" * 100)
spy = yf.download("SPY", start="1999-03-10", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
upro = yf.download("UPRO", start="2009-06-25", end=END, auto_adjust=True, progress=False)["Close"].squeeze()

for name, px in [("SPY (1x S&P)", spy), ("UPRO (3x S&P)", upro)]:
    eq, ret, n = backtest_clean(px, px, 5, 200)
    s_cagr, s_mdd, s_sh, s_cal, fv = metrics(eq, ret)
    bh_eq = (px / px.iloc[0]) * INIT_CASH
    bh_ret = bh_eq.pct_change().fillna(0)
    b_cagr, b_mdd, b_sh, b_cal, _ = metrics(bh_eq, bh_ret)
    print(f"\n{name} ({px.index[0].date()} ~ {px.index[-1].date()})")
    print(f"  EMA5/200:    CAGR {s_cagr*100:>6.2f}%, MDD {s_mdd*100:>6.2f}%, Calmar {s_cal:.3f}, Trades {n}")
    print(f"  Buy & Hold:  CAGR {b_cagr*100:>6.2f}%, MDD {b_mdd*100:>6.2f}%, Calmar {b_cal:.3f}")

print()
print("[6/6] 6 practical anti-overfitting optimizations (executable versions)")
print("=" * 100)

# Optimization 1: VIX filter (using ^VIX)
print("\n[Optimization 1] VIX filter - force flat when VIX > 30")
vix = yf.download("^VIX", start="1999-03-10", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
vix = vix.reindex(tqqq_combined.index).ffill()

def backtest_with_vix(px, fast, slow, vix_thresh=30):
    ema_f = px.ewm(span=fast, adjust=False).mean()
    ema_s = px.ewm(span=slow, adjust=False).mean()
    sig = ((ema_f > ema_s) & (vix < vix_thresh)).astype(int)
    sig.iloc[:slow] = 0
    pos = sig.shift(1).fillna(0)
    daily_ret = px.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * 7.5 / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret, int((pos.diff().abs() > 0).sum())

eq, ret, n = backtest_with_vix(tqqq_combined, 5, 200, 30)
c, m, sh, cal, fv = metrics(eq, ret)
print(f"  Full sample 1999-2026: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {cal:.3f}, Trades {n}")

# Optimization 2: position sizing instead of 100%/0%
print("\n[Optimization 2] Partial sizing (70% position instead of 100%) to reduce TQQQ leverage exposure")
def backtest_partial(px, fast, slow, pos_size=0.7):
    ema_f = px.ewm(span=fast, adjust=False).mean()
    ema_s = px.ewm(span=slow, adjust=False).mean()
    sig = (ema_f > ema_s).astype(float) * pos_size
    sig.iloc[:slow] = 0
    pos = sig.shift(1).fillna(0)
    daily_ret = px.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * 7.5 / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret

eq, ret = backtest_partial(tqqq_combined, 5, 200, 0.7)
c, m, sh, cal, fv = metrics(eq, ret)
print(f"  70% position: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {cal:.3f}")

# Optimization 3: ensemble of multiple parameters
print("\n[Optimization 3] Multi-EMA ensemble (5/100 + 5/200 + 10/200, each 1/3 weight)")
def backtest_ensemble(px, params_list):
    sigs = []
    for f, s in params_list:
        ema_f = px.ewm(span=f, adjust=False).mean()
        ema_s = px.ewm(span=s, adjust=False).mean()
        sig = (ema_f > ema_s).astype(float)
        sig.iloc[:s] = 0
        sigs.append(sig)
    avg_sig = sum(sigs) / len(sigs)
    pos = avg_sig.shift(1).fillna(0)
    daily_ret = px.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * 7.5 / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret

eq, ret = backtest_ensemble(tqqq_combined, [(5, 100), (5, 200), (10, 200)])
c, m, sh, cal, fv = metrics(eq, ret)
print(f"  3-signal ensemble: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {cal:.3f}")
