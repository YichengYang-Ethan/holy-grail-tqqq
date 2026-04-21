"""
WF rotation + cash reserve pyramid dip-buy (per user's correct interpretation)

Design:
- Bull (EMA signal up): 100% TQQQ
- Bear (EMA signal down): X% QQQ + (100-X)% cash reserve (standing by to buy dips)
- The deeper below MA200, the more of the cash reserve is deployed to TQQQ:
  - MA200 -10%: deploy 25% of cash reserve -> TQQQ
  - MA200 -20%: deploy 50% of cash reserve -> TQQQ
  - MA200 -30%: deploy 75% of cash reserve -> TQQQ
  - MA200 -40%: deploy 100% of cash reserve -> TQQQ
- Signal flips back to bull (EMA cross up): sell QQQ + sell dip-buy TQQQ -> 100% TQQQ
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0

print("[1/5] Data preparation ...")
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
print(f"  QQQ {qqq.index[0].date()} ~ {qqq.index[-1].date()}, {len(qqq)} days")
print(f"  TQQQ start ${tqqq_full.iloc[0]:.4f}, end ${tqqq_full.iloc[-1]:.2f}")

def backtest_3asset(qpos, tpos, cpos, qqq, tqqq, fee_bps=2.5, slip_bps=5.0):
    """
    qpos + tpos + cpos = 1.0 each day. cpos = cash weight (0% interest)
    """
    qpos_lag = qpos.shift(1).fillna(0)
    tpos_lag = tpos.shift(1).fillna(0)
    qret = qqq.pct_change().fillna(0)
    tret = tqqq.pct_change().fillna(0)
    pos_change = qpos_lag.diff().abs().fillna(0) + tpos_lag.diff().abs().fillna(0)
    cost = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = qpos_lag * qret + tpos_lag * tret - cost
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    n_trades = int((pos_change > 0.05).sum())
    return eq, strat_ret, n_trades

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

def wf_rotation_pyramid(qqq, tqqq_full, train_years=5, test_years=2,
                        bear_qqq_pct=0.60, bear_cash_pct=0.40,
                        pyramid_levels=None, ma_for_pyramid=200):
    """
    Walk-forward rotation + cash-reserve pyramid
    - bear_qqq_pct: bear QQQ share
    - bear_cash_pct: bear cash-reserve share (for pyramid dip-buy)
    - pyramid_levels: [(drawdown_threshold, deploy_fraction)]
    """
    if pyramid_levels is None:
        pyramid_levels = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]

    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    chosen_params = []
    test_periods = []

    start_idx = 252 * train_years
    while start_idx + 252 * test_years <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_years, len(qqq))
        train_idx = qqq.index[train_end - 252*train_years : train_end]
        test_idx = qqq.index[train_end : test_end]

        # Pick best (f, s) on training set
        best_cal = -999
        best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                ema_f = qqq.loc[train_idx].ewm(span=f, adjust=False).mean()
                ema_s = qqq.loc[train_idx].ewm(span=s, adjust=False).mean()
                bull = (ema_f > ema_s)
                bull_arr = bull.astype(float); bull_arr.iloc[:s] = 0
                # Simplified: use 100% TQQQ / 100% QQQ for param selection
                eq, ret, _ = backtest_3asset(
                    1 - bull_arr, bull_arr, pd.Series(0.0, index=train_idx),
                    qqq.loc[train_idx], tqqq_full.loc[train_idx]
                )
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)

        # Apply to test period + pyramid logic
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_years : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        ma_p = qqq.loc[full_idx].ewm(span=ma_for_pyramid, adjust=False).mean()

        bull = (ema_f > ema_s)
        bull.iloc[:s] = False

        # Compute daily positions
        qpos = pd.Series(0.0, index=full_idx)
        tpos = pd.Series(0.0, index=full_idx)
        cpos = pd.Series(0.0, index=full_idx)

        for i in range(len(full_idx)):
            t = full_idx[i]
            if i < max(s, ma_for_pyramid):
                cpos.iloc[i] = 1.0
                continue

            if bull.iloc[i]:
                # Bull: 100% TQQQ
                qpos.iloc[i] = 0
                tpos.iloc[i] = 1.0
                cpos.iloc[i] = 0
            else:
                # Bear: bear_qqq_pct QQQ + bear_cash_pct cash (ready for pyramid dip-buy)
                # Compute price deviation from MA200
                deviation = float(qqq.loc[t] / ma_p.loc[t] - 1)
                # Fraction of cash to deploy (by pyramid tier)
                deployed_frac = 0.0
                for thresh, frac in pyramid_levels:
                    if deviation <= thresh:
                        deployed_frac = frac
                # Deployed cash -> TQQQ
                deployed_cash = bear_cash_pct * deployed_frac
                tpos.iloc[i] = deployed_cash
                cpos.iloc[i] = bear_cash_pct - deployed_cash
                qpos.iloc[i] = bear_qqq_pct

        # Extract for test period
        qpos_t = qpos.loc[test_idx]
        tpos_t = tpos.loc[test_idx]
        cpos_t = cpos.loc[test_idx]

        eq, ret, _ = backtest_3asset(
            qpos_t, tpos_t, cpos_t,
            qqq.loc[test_idx], tqqq_full.loc[test_idx]
        )
        all_returns.append(ret)
        chosen_params.append(best_p)
        test_periods.append((test_idx[0], test_idx[-1]))
        start_idx += 252 * test_years

    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen_params, test_periods


print()
print("=" * 100)
print("[Experiment] WF rotation + cash-reserve pyramid - multi-config comparison")
print("=" * 100)

# Control: previous best WF rotation
def wf_pure_rotation(qqq, tqqq_full, train_years=5, test_years=2):
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
                bull = (ema_f > ema_s).astype(float); bull.iloc[:s] = 0
                eq, ret, _ = backtest_3asset(1 - bull, bull, pd.Series(0.0, index=train_idx), qqq.loc[train_idx], tqqq_full.loc[train_idx])
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_years : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        bull = (ema_f > ema_s).astype(float); bull.iloc[:s] = 0
        bull_t = bull.loc[test_idx]
        eq, ret, _ = backtest_3asset(1 - bull_t, bull_t, pd.Series(0.0, index=test_idx), qqq.loc[test_idx], tqqq_full.loc[test_idx])
        all_returns.append(ret)
        start_idx += 252 * test_years
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret

print("\nBaseline (previous best WF QQQ/TQQQ rotation) ...")
base_eq, base_ret = wf_pure_rotation(qqq, tqqq_full)
c, m, sh, ca, fv, so = metrics(base_eq, base_ret)
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}, Final ${fv:,.0f}")

# Multiple pyramid configurations
configs = {
    "P1 60Q/40C, dip-buy -10/-20/-30/-40 -> 25/50/75/100%": (
        0.60, 0.40, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P2 70Q/30C, dip-buy -10/-20/-30/-40 -> 25/50/75/100%": (
        0.70, 0.30, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P3 50Q/50C, dip-buy -10/-20/-30/-40 -> 25/50/75/100%": (
        0.50, 0.50, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P4 60Q/40C, aggressive -5/-15/-25/-35 -> 25/50/75/100%": (
        0.60, 0.40, [(-0.05, 0.25), (-0.15, 0.50), (-0.25, 0.75), (-0.35, 1.00)], 200),
    "P5 60Q/40C, conservative -15/-25/-35/-45 -> 25/50/75/100%": (
        0.60, 0.40, [(-0.15, 0.25), (-0.25, 0.50), (-0.35, 0.75), (-0.45, 1.00)], 200),
    "P6 60Q/40C, early deploy -5/-10/-20/-30 -> 25/50/75/100%": (
        0.60, 0.40, [(-0.05, 0.25), (-0.10, 0.50), (-0.20, 0.75), (-0.30, 1.00)], 200),
    "P7 80Q/20C, dip-buy -10/-20/-30/-40 -> 25/50/75/100%": (
        0.80, 0.20, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P8 0Q/100C, all-cash standby (extreme)": (
        0.0, 1.0, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P9 60Q/40C, simplified -20/-40 -> 50/100%": (
        0.60, 0.40, [(-0.20, 0.50), (-0.40, 1.00)], 200),
    "P10 60Q/40C, 5 tiers -10/-20/-30/-40/-50 -> 20/40/60/80/100%": (
        0.60, 0.40, [(-0.10, 0.20), (-0.20, 0.40), (-0.30, 0.60), (-0.40, 0.80), (-0.50, 1.00)], 200),
}

print(f"\n{'Pyramid config':<58} {'CAGR':>7} {'MDD':>9} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'Final Value':>14}")
print("-" * 130)

results = {}
for name, (qp, cp, levels, ma) in configs.items():
    eq, ret, params, periods = wf_rotation_pyramid(
        qqq, tqqq_full, 5, 2,
        bear_qqq_pct=qp, bear_cash_pct=cp,
        pyramid_levels=levels, ma_for_pyramid=ma
    )
    c, m, sh, ca, fv, so = metrics(eq, ret)
    results[name] = (c, m, sh, ca, fv, so, eq, ret)
    print(f"{name:<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# Add baseline
print("-" * 130)
c, m, sh, ca, fv, so = metrics(base_eq, base_ret)
print(f"{'Baseline: WF QQQ/TQQQ rotation (no pyramid)':<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# Same-period buy & hold
test_period = base_ret.index
qqq_t = qqq.loc[test_period]
tqqq_t = tqqq_full.loc[test_period]
for name, px in [("TQQQ B&H same period", tqqq_t), ("QQQ B&H same period", qqq_t)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# Find best
print()
print("=" * 100)
print("[Champion analysis]")
print("=" * 100)
best_by_calmar = max(results.items(), key=lambda x: x[1][3])
best_by_cagr = max(results.items(), key=lambda x: x[1][0])
best_by_sharpe = max(results.items(), key=lambda x: x[1][2])

print(f"\nHighest Calmar: {best_by_calmar[0]}")
print(f"  CAGR {best_by_calmar[1][0]*100:.2f}%, MDD {best_by_calmar[1][1]*100:.2f}%, Calmar {best_by_calmar[1][3]:.3f}")

print(f"\nHighest CAGR:   {best_by_cagr[0]}")
print(f"  CAGR {best_by_cagr[1][0]*100:.2f}%, MDD {best_by_cagr[1][1]*100:.2f}%, Calmar {best_by_cagr[1][3]:.3f}")

print(f"\nHighest Sharpe: {best_by_sharpe[0]}")
print(f"  CAGR {best_by_sharpe[1][0]*100:.2f}%, MDD {best_by_sharpe[1][1]*100:.2f}%, Sharpe {best_by_sharpe[1][2]:.3f}")

# Key comparison
print()
print("=" * 100)
print("[vs baseline] Best pyramid vs WF pure rotation")
print("=" * 100)
c0, m0, sh0, ca0, fv0, so0 = metrics(base_eq, base_ret)
cb, mb, shb, cab, fvb, sob = best_by_calmar[1][:6]
print(f"\nBaseline WF rotation:                       CAGR {c0*100:.2f}%, MDD {m0*100:.2f}%, Sharpe {sh0:.3f}, Calmar {ca0:.3f}, Final ${fv0:,.0f}")
print(f"Best pyramid ({best_by_calmar[0][:30]}...):")
print(f"                                    CAGR {cb*100:.2f}%, MDD {mb*100:.2f}%, Sharpe {shb:.3f}, Calmar {cab:.3f}, Final ${fvb:,.0f}")
print(f"\nChange:  CAGR {(cb-c0)*100:+.2f}pp,  MDD {(mb-m0)*100:+.2f}pp,  Calmar {cab-ca0:+.3f},  Final {(fvb/fv0-1)*100:+.1f}%")

# Bootstrap significance
print()
print("=" * 100)
print("[Bootstrap 1000 resamples] Best pyramid significance")
print("=" * 100)

best_ret = best_by_calmar[1][7]
np.random.seed(42)
cagrs = []; sharpes = []
n_obs = len(best_ret); block = 20
for _ in range(1000):
    starts = np.random.randint(0, n_obs - block, n_obs // block)
    idx = np.concatenate([np.arange(st, st+block) for st in starts])
    idx = idx[idx < n_obs]
    s = best_ret.values[idx]
    eq = np.cumprod(1 + s)
    years = len(s) / 252
    if eq[-1] > 0: cagrs.append(eq[-1] ** (1/years) - 1)
    if s.std() > 0: sharpes.append(s.mean() * 252 / (s.std() * np.sqrt(252)))
cagrs, sharpes = np.array(cagrs), np.array(sharpes)
print(f"\n{best_by_calmar[0]}:")
print(f"  CAGR  median {np.median(cagrs)*100:.2f}%, 95% CI [{np.percentile(cagrs, 2.5)*100:.2f}%, {np.percentile(cagrs, 97.5)*100:.2f}%]")
print(f"  Sharpe median {np.median(sharpes):.3f}, 95% CI [{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]")
print(f"  P(CAGR > 0): {(cagrs > 0).mean()*100:.1f}%")
print(f"  P(CAGR > 20%): {(cagrs > 0.20).mean()*100:.1f}%")
print(f"  P(CAGR > 25%): {(cagrs > 0.25).mean()*100:.1f}%")
