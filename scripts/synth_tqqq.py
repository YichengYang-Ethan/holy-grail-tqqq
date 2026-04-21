"""
Synthetic TQQQ 1999-2026 data + full-sample backtest + parameter stability analysis

Synthetic formula: TQQQ_ret = 3 * QQQ_ret - daily_cost
- expense ratio: 0.84%/year
- financing: ~2 * short-term rate
- Verification: compare synthetic 2010-2026 vs real TQQQ to see if they match
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "1999-03-10"  # QQQ listing date
END = "2026-01-17"
INIT_CASH = 10_000.0

print("[1/5] Downloading full-history QQQ + real TQQQ ...")
qqq_full = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
# Pull short-term rate proxy (DTB3 = 3-month T-bill, or use SHV ETF as proxy; here use approximate constant)
# Simplified: 2000-2008 avg ~3%, 2009-2015 ~0.1%, 2016-2019 ~1.5%, 2020-2021 ~0.1%, 2022-2026 ~4.5%
def daily_financing(date):
    y = date.year
    if y <= 2007: rate = 0.04
    elif y <= 2008: rate = 0.02
    elif y <= 2015: rate = 0.001
    elif y <= 2019: rate = 0.015
    elif y <= 2021: rate = 0.001
    else: rate = 0.045
    # 2x borrowing + 40bp spread
    return (rate + 0.004) * 2 / 252

print("[2/5] Building synthetic TQQQ ...")
qqq_ret = qqq_full.pct_change().fillna(0)
expense_daily = 0.0084 / 252
financing_daily = pd.Series([daily_financing(d) for d in qqq_full.index], index=qqq_full.index)
slip_daily = 0.0005  # 5 bp/day rebalancing slippage

tqqq_synth_ret = 3 * qqq_ret - expense_daily - financing_daily - slip_daily / 252 * 252  # = - slip_daily
# Initial price on day 1 = $1
tqqq_synth = (1 + tqqq_synth_ret).cumprod()

print("[3/5] Validating synthetic vs real TQQQ (2010-2026 overlap period) ...")
overlap_start = tqqq_real.index[0]
synth_overlap = tqqq_synth.loc[overlap_start:]
real_overlap = tqqq_real.loc[overlap_start:]
# Align starting point, normalize
synth_norm = synth_overlap / synth_overlap.iloc[0]
real_norm = real_overlap / real_overlap.iloc[0]

# Compare final values
final_synth = float(synth_norm.iloc[-1])
final_real = float(real_norm.iloc[-1])
print(f"  Synthetic TQQQ {overlap_start.date()}->now: {final_synth:.1f}x")
print(f"  Real TQQQ {overlap_start.date()}->now: {final_real:.1f}x")
print(f"  Error: {abs(final_synth - final_real)/final_real*100:.1f}%")

# Splice synthetic + real: use synthetic for 1999-2010, real from 2010+ (aligned to real's start)
scale = float(real_overlap.iloc[0]) / float(tqqq_synth.loc[overlap_start])
tqqq_pre2010 = tqqq_synth.loc[:overlap_start] * scale
tqqq_combined = pd.concat([tqqq_pre2010.iloc[:-1], tqqq_real])
print(f"  Spliced TQQQ series: {tqqq_combined.index[0].date()} ~ {tqqq_combined.index[-1].date()}, {len(tqqq_combined)} days")

print()
print("[4/5] Backtest: EMA5/200 TQQQ-sig -> close[T+1] on 1999-2026 full sample")
print("=" * 100)

def backtest_clean(close_sig, close_exec, fast, slow, fee_bps=2.5, slip_bps=5.0):
    ema_f = close_sig.ewm(span=fast, adjust=False).mean()
    ema_s = close_sig.ewm(span=slow, adjust=False).mean()
    sig = (ema_f > ema_s).astype(int)
    sig.iloc[:slow] = 0
    pos = sig.shift(1).fillna(0)  # next-day position
    daily_ret = close_exec.pct_change().fillna(0)
    # Transaction cost: charge (fee + slip) bps on each position change
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    n_trades = int((pos.diff().abs() > 0).sum())
    return eq, strat_ret, n_trades

def metrics(eq, ret, label):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {"label": label, "cagr": cagr, "mdd": mdd, "sharpe": sh, "calmar": cal, "fv": eq.iloc[-1]}

# Full sample 1999-2026 uses synthetic + real splice
periods = {
    "Full sample 1999-2026 (dot-com + 2008 + COVID + 2022)": tqqq_combined,
    "Out-of-sample 1999-2010 (dot-com + 2008)": tqqq_combined.loc[:"2010-02-10"],
    "In-sample 2010-2026 (real TQQQ)": tqqq_real,
}

# Compare with buy & hold
def bh_metrics(price, label):
    eq = (price / price.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    return metrics(eq, ret, label)

results_by_period = {}
for period_name, px in periods.items():
    print(f"\n--- {period_name} ---")
    print(f"{'Strategy':<40} {'CAGR':>9} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'Final Value':>14} {'Trades':>5}")
    print("-" * 100)
    eq, ret, n = backtest_clean(px, px, 5, 200)
    m = metrics(eq, ret, "EMA5/200 self-sig"); m['n'] = n
    print(f"{m['label']:<40} {m['cagr']*100:>8.2f}% {m['mdd']*100:>8.2f}% {m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['fv']:>14,.0f} {n:>5}")
    bh = bh_metrics(px, "TQQQ Buy & Hold")
    print(f"{bh['label']:<40} {bh['cagr']*100:>8.2f}% {bh['mdd']*100:>8.2f}% {bh['sharpe']:>8.3f} {bh['calmar']:>8.3f} {bh['fv']:>14,.0f} {'-':>5}")
    results_by_period[period_name] = (m, bh)

print()
print("[5/5] Parameter stability scan: fast * slow grid (key anti-overfitting test)")
print("=" * 100)
print("\nUsing full sample 1999-2026 (key: only parameters stable out-of-sample are good ones)")

fast_grid = [3, 5, 8, 10, 13, 20]
slow_grid = [50, 100, 150, 200, 250, 300]

print(f"\n{'CAGR%':<8}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
for f in fast_grid:
    print(f"f={f:<5}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        try:
            eq, ret, n = backtest_clean(tqqq_combined, tqqq_combined, f, s)
            m = metrics(eq, ret, "")
            print(f"{m['cagr']*100:>8.1f}%", end="")
        except Exception as e:
            print(f"{'err':>9}", end="")
    print()

print(f"\n{'MDD%':<8}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
for f in fast_grid:
    print(f"f={f:<5}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        try:
            eq, ret, n = backtest_clean(tqqq_combined, tqqq_combined, f, s)
            m = metrics(eq, ret, "")
            print(f"{m['mdd']*100:>8.1f}%", end="")
        except: pass
    print()

print(f"\n{'Calmar':<8}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
for f in fast_grid:
    print(f"f={f:<5}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        try:
            eq, ret, n = backtest_clean(tqqq_combined, tqqq_combined, f, s)
            m = metrics(eq, ret, "")
            print(f"{m['calmar']:>9.3f}", end="")
        except: pass
    print()

print()
print("=" * 100)
print("[Key interpretation]")
print("=" * 100)
print("- Look at CAGR/Calmar tables: ideal parameters should sit on a plateau, not an isolated peak")
print("- If a combination has very high CAGR but all neighbors are poor -> overfitting")
print("- Truly robust parameters: neighbors (f +/-2, s +/-50) are also decent")
