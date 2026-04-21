"""
Fully adaptive Walk-Forward + retrain frequency study
- Adaptive: (fast, slow) + pyramid parameters (base_tqqq, anchor, levels)
- Frequency study: training window length + test window length + event-driven retraining
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0

print("[1/5] Data preparation ...")
qqq = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

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
vix_a = vix.reindex(qqq.index).ffill()
print(f"  Data range: {qqq.index[0].date()} ~ {qqq.index[-1].date()}, {len(qqq)} days")

# ============================================================
def get_bull(qqq, fast, slow):
    ema_f = qqq.ewm(span=fast, adjust=False).mean()
    ema_s = qqq.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s)
    bull.iloc[:slow] = False
    return bull

def build_position(qqq, bull, ma_period, base_tqqq, levels, anchor):
    """Generate TQQQ position series"""
    if anchor == "ma":
        ma = qqq.ewm(span=ma_period, adjust=False).mean()
        deviation = qqq / ma - 1
    else:  # peak
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

def backtest(tpos, tqqq, fee_bps=2.5, slip_bps=5.0):
    tpos_lag = tpos.shift(1).fillna(0)
    tret = tqqq.pct_change().fillna(0)
    pos_change = tpos_lag.diff().abs().fillna(0)
    cost = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = tpos_lag * tret - cost
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1 if eq.iloc[-1] > 0 else -1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 and cagr > 0 else (cagr if mdd == 0 else cagr / abs(mdd))
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

# ============================================================
# Fully adaptive Walk-Forward
# ============================================================
def fully_adaptive_wf(qqq, tqqq_full, train_y=5, test_y=2, search_grid=None, verbose=False):
    """
    walk-forward jointly optimizing:
    - (fast, slow) EMA signal
    - base_tqqq (bear-market base position)
    - anchor (ma vs peak)
    - pyramid levels
    """
    if search_grid is None:
        search_grid = {
            'fast_slow': [(3,100),(5,100),(5,150),(5,200),(5,250),(8,100),(8,150),(8,200),(10,150),(10,200),(10,250),(13,200)],
            'base_tqqq': [0.0, 0.25, 0.50, 0.75],
            'anchor': ['ma', 'peak'],
            'levels': [
                [(-0.05,0.25),(-0.10,0.50),(-0.20,0.75),(-0.30,1.00)],  # early + aggressive
                [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)],  # standard
                [(-0.15,0.25),(-0.25,0.50),(-0.35,0.75),(-0.45,1.00)],  # conservative
                [(-0.05,0.50),(-0.20,1.00)],                             # simple 2-level
                [(-0.10,0.33),(-0.20,0.66),(-0.30,1.00)],                # 3-level
            ],
        }

    all_returns = []
    chosen_params = []
    test_periods = []

    start_idx = 252 * train_y
    while start_idx + 252 * test_y <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_y, len(qqq))
        train_idx = qqq.index[train_end - 252*train_y : train_end]
        test_idx = qqq.index[train_end : test_end]

        # Grid search
        best_cal = -999
        best_params = None
        for (fast, slow) in search_grid['fast_slow']:
            for base in search_grid['base_tqqq']:
                for anchor in search_grid['anchor']:
                    for levels in search_grid['levels']:
                        bull_t = get_bull(qqq.loc[train_idx], fast, slow)
                        tpos = build_position(qqq.loc[train_idx], bull_t, slow, base, levels, anchor)
                        eq, ret = backtest(tpos, tqqq_full.loc[train_idx])
                        c, m, sh, cal, _, _ = metrics(eq, ret)
                        if cal > best_cal:
                            best_cal = cal
                            best_params = (fast, slow, base, anchor, levels)

        # Apply to test period
        fast, slow, base, anchor, levels = best_params
        full_idx = qqq.index[train_end - 252*train_y : test_end]
        bull_t = get_bull(qqq.loc[full_idx], fast, slow)
        tpos = build_position(qqq.loc[full_idx], bull_t, slow, base, levels, anchor)
        tpos_test = tpos.loc[test_idx]
        eq, ret = backtest(tpos_test, tqqq_full.loc[test_idx])
        all_returns.append(ret)
        chosen_params.append(best_params)
        test_periods.append((test_idx[0], test_idx[-1]))
        start_idx += 252 * test_y

        if verbose:
            print(f"    {test_idx[0].date()}~{test_idx[-1].date()}: "
                  f"f={fast},s={slow},base={base:.2f},anchor={anchor},"
                  f"levels={[(t,round(f,2)) for t,f in levels]}")

    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen_params, test_periods


# ============================================================
# Experiment 1: baseline fully adaptive (5y/2y)
# ============================================================
print()
print("=" * 100)
print("[Experiment 1] Fully adaptive WF (train 5y, test 2y) - all parameters adaptive")
print("=" * 100)
print("\nSearch space:")
print("  - (fast, slow): 12 combinations")
print("  - base_tqqq: [0, 0.25, 0.50, 0.75]")
print("  - anchor: [ma, peak]")
print("  - levels: 5 pyramid configurations")
print("  - Total: 12 * 4 * 2 * 5 = 480 combinations per training window")
print()

print("Running (about 3-5 minutes) ...")
eq_full, ret_full, params_full, periods_full = fully_adaptive_wf(qqq, tqqq_full, 5, 2, verbose=True)
c, m, sh, ca, fv, so = metrics(eq_full, ret_full)
print(f"\nFully adaptive WF results:")
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}")
print(f"  Final value ${fv:,.0f}, $150K -> ${fv*15:,.0f}")

# ============================================================
# Experiment 2: train/test window length scan
# ============================================================
print()
print("=" * 100)
print("[Experiment 2] Train / test window length scan - find best retrain frequency")
print("=" * 100)

# Use a smaller search space for speed
small_grid = {
    'fast_slow': [(3,200),(5,150),(5,200),(8,200),(10,200),(13,200)],
    'base_tqqq': [0.0, 0.50],
    'anchor': ['ma', 'peak'],
    'levels': [
        [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)],
        [(-0.05,0.50),(-0.20,1.00)],
    ],
}

window_combos = [
    (3, 1), (3, 2), (3, 3),
    (5, 1), (5, 2), (5, 3), (5, 5),
    (7, 1), (7, 2), (7, 3),
    (10, 1), (10, 2), (10, 3),
]

print(f"\n{'Train x Test':<14} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'# Changes':>10} {'Final Value':>14}")
print("-" * 100)
window_results = {}
for tr, te in window_combos:
    if 252 * tr + 252 * te > len(qqq) - 252:
        continue
    eq, ret, ps, pers = fully_adaptive_wf(qqq, tqqq_full, tr, te, search_grid=small_grid, verbose=False)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    window_results[(tr, te)] = (c, m, sh, ca, fv, so, eq, ret)
    print(f"{tr}y train x {te}y test {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {len(ps):>10} {fv:>14,.0f}")

# Find best window
best_window = max(window_results.items(), key=lambda x: x[1][3])
print(f"\nBest window: train={best_window[0][0]}y, test={best_window[0][1]}y -> Calmar {best_window[1][3]:.3f}")

# ============================================================
# Experiment 3: event-driven retraining - triggered by VIX / drawdown
# ============================================================
print()
print("=" * 100)
print("[Experiment 3] Event-driven retraining - retrain immediately when regime shifts")
print("=" * 100)

def event_driven_wf(qqq, tqqq_full, vix_a, train_y=5, max_test_y=5,
                     min_test_d=60, vix_thresh=35, dd_thresh=-0.20,
                     search_grid=None, verbose=False):
    """
    Event-driven retraining:
    - Must hold at least min_test_d days before retrain
    - Trigger (any one):
      * max_test_y years reached since last retrain
      * VIX breaks through vix_thresh (regime shift)
      * Cumulative drawdown < dd_thresh
    """
    if search_grid is None:
        search_grid = small_grid

    all_returns = []
    chosen_params = []
    test_periods = []

    current_idx = 252 * train_y
    while current_idx + min_test_d <= len(qqq):
        train_end = current_idx
        train_start = max(0, train_end - 252*train_y)
        train_idx = qqq.index[train_start:train_end]

        # Find next retrain point
        # First determine farthest position max_test_d days later
        max_test_d = 252 * max_test_y
        scan_end = min(current_idx + max_test_d, len(qqq))

        # Within [current_idx + min_test_d, scan_end), find earliest trigger condition
        retrain_idx = scan_end  # default expiration
        equity_proxy = (qqq / qqq.iloc[max(0, current_idx-1)] - 1)  # simplified drawdown proxy
        for j in range(current_idx + min_test_d, scan_end):
            v = vix_a.iloc[j]
            # Compute cumulative drawdown within test period
            test_slice = qqq.iloc[current_idx:j+1]
            mdd_so_far = (test_slice / test_slice.cummax() - 1).min()
            if (not pd.isna(v) and v > vix_thresh and j > current_idx + min_test_d) or mdd_so_far < dd_thresh:
                retrain_idx = j
                break

        test_end = retrain_idx
        test_idx = qqq.index[current_idx:test_end]

        # Train: find best parameters
        best_cal = -999
        best_params = None
        for (fast, slow) in search_grid['fast_slow']:
            for base in search_grid['base_tqqq']:
                for anchor in search_grid['anchor']:
                    for levels in search_grid['levels']:
                        bull_t = get_bull(qqq.loc[train_idx], fast, slow)
                        tpos = build_position(qqq.loc[train_idx], bull_t, slow, base, levels, anchor)
                        eq, ret = backtest(tpos, tqqq_full.loc[train_idx])
                        c, m, sh, cal, _, _ = metrics(eq, ret)
                        if cal > best_cal:
                            best_cal = cal
                            best_params = (fast, slow, base, anchor, levels)

        # Apply
        fast, slow, base, anchor, levels = best_params
        full_idx = qqq.index[train_start:test_end]
        bull_t = get_bull(qqq.loc[full_idx], fast, slow)
        tpos = build_position(qqq.loc[full_idx], bull_t, slow, base, levels, anchor)
        tpos_test = tpos.loc[test_idx]
        eq, ret = backtest(tpos_test, tqqq_full.loc[test_idx])
        all_returns.append(ret)
        chosen_params.append(best_params)
        test_periods.append((test_idx[0], test_idx[-1]))
        current_idx = test_end

    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen_params, test_periods

print("\nRunning event-driven WF (VIX>35 or drawdown <-20% triggers, max 5y between retrains) ...")
event_eq, event_ret, event_params, event_periods = event_driven_wf(
    qqq, tqqq_full, vix_a, train_y=5, max_test_y=5,
    min_test_d=126, vix_thresh=35, dd_thresh=-0.20
)
c, m, sh, ca, fv, so = metrics(event_eq, event_ret)
print(f"\nEvent-driven WF results:")
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}")
print(f"  Final value ${fv:,.0f}")
print(f"  Retrained {len(event_periods)} times")
print(f"\nRetraining windows:")
for i, (sd, ed) in enumerate(event_periods):
    days = (ed - sd).days
    print(f"  #{i+1}: {sd.date()} -> {ed.date()} ({days} days)")

# Scan multiple event trigger thresholds
print()
print("--- Different VIX threshold scan ---")
print(f"\n{'VIX thresh':<10} {'DD thresh':<10} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'# Retrains':>10}")
print("-" * 80)
for vix_t in [25, 30, 35, 40, 50]:
    for dd_t in [-0.10, -0.20, -0.30]:
        eq, ret, ps, pers = event_driven_wf(qqq, tqqq_full, vix_a, 5, 5, 126, vix_t, dd_t)
        c, m, sh, ca, fv, so = metrics(eq, ret)
        print(f"VIX>{vix_t:<6} DD<{dd_t*100:>4.0f}% {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {len(pers):>10}")

# ============================================================
# Experiment 4: ultimate comparison
# ============================================================
print()
print("=" * 100)
print("[Final comparison] All WF strategies + benchmark")
print("=" * 100)

# Baseline
def wf_qqq_tqqq(qqq, tqqq_full, train_y=5, test_y=2):
    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    start_idx = 252 * train_y
    while start_idx + 252 * test_y <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_y, len(qqq))
        train_idx = qqq.index[train_end - 252*train_y : train_end]
        test_idx = qqq.index[train_end : test_end]
        best_cal = -999; best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                bull_t = get_bull(qqq.loc[train_idx], f, s).astype(float)
                qpos = (1 - bull_t); tpos = bull_t
                qret = qqq.loc[train_idx].pct_change().fillna(0)
                tret = tqqq_full.loc[train_idx].pct_change().fillna(0)
                qpos_l = qpos.shift(1).fillna(0); tpos_l = tpos.shift(1).fillna(0)
                cost = (qpos_l.diff().abs().fillna(0) + tpos_l.diff().abs().fillna(0)) * 7.5/10000
                strat_ret = qpos_l * qret + tpos_l * tret - cost
                eq = (1 + strat_ret).cumprod() * INIT_CASH
                _, _, _, cal, _, _ = metrics(eq, strat_ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_y : test_end]
        bull_t = get_bull(qqq.loc[full_idx], f, s).astype(float)
        bull_test = bull_t.loc[test_idx]
        qpos = 1 - bull_test; tpos = bull_test
        qret = qqq.loc[test_idx].pct_change().fillna(0); tret = tqqq_full.loc[test_idx].pct_change().fillna(0)
        qpos_l = qpos.shift(1).fillna(0); tpos_l = tpos.shift(1).fillna(0)
        cost = (qpos_l.diff().abs().fillna(0) + tpos_l.diff().abs().fillna(0)) * 7.5/10000
        strat_ret = qpos_l * qret + tpos_l * tret - cost
        all_returns.append(strat_ret)
        start_idx += 252 * test_y
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret

print("\nRunning WF QQQ/TQQQ rotation benchmark ...")
base_eq, base_ret = wf_qqq_tqqq(qqq, tqqq_full)

print(f"\n{'Method':<46} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Final Value':>14}")
print("-" * 130)

methods = []
c, m, sh, ca, fv, so = metrics(eq_full, ret_full)
methods.append(("Fully adaptive WF (5y/2y, 480 combos)", c, m, sh, so, ca, fv))
c, m, sh, ca, fv, so = metrics(event_eq, event_ret)
methods.append(("Event-driven WF (VIX>35, DD<-20%)", c, m, sh, so, ca, fv))
c, m, sh, ca, fv, so = metrics(base_eq, base_ret)
methods.append(("Benchmark: WF QQQ/TQQQ rotation (5y/2y)", c, m, sh, so, ca, fv))

# Best window from experiment 2
(tr, te), (c, m, sh, ca, fv, so, eq_w, ret_w) = best_window
methods.append((f"Best window ({tr}y/{te}y) fully adaptive", c, m, sh, so, ca, fv))

# B&H same period
test_period = base_ret.index
for name, px in [("TQQQ B&H same period", tqqq_full.loc[test_period]), ("QQQ B&H same period", qqq.loc[test_period])]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    methods.append((name, c, m, sh, so, ca, fv))

# Sort by Calmar
methods.sort(key=lambda x: -x[5])
for name, c, m, sh, so, ca, fv in methods:
    print(f"{name:<46} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}")

# 1999-2010 OOS test
print()
print("=" * 100)
print("[Overfitting check] Best fully adaptive strategy on 1999-2010 OOS")
print("=" * 100)

# Use 1999-2010 data fully independently to test fully adaptive version
oos_qqq = qqq.loc[:"2010-02-10"]
oos_tqqq = tqqq_full.loc[:"2010-02-10"]
print("\nRun fully adaptive independently on 1999-2010 data (fully out-of-sample) ...")
if len(oos_qqq) > 252 * 7:
    oos_eq, oos_ret, oos_p, _ = fully_adaptive_wf(oos_qqq, oos_tqqq, 5, 2, search_grid=small_grid)
    c, m, sh, ca, fv, so = metrics(oos_eq, oos_ret)
    print(f"  OOS 1999-2010: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {ca:.3f}")

print()
print("=" * 100)
print("[Champion parameter trajectory] Check parameters chosen in each test window")
print("=" * 100)
for i, ((sd, ed), p) in enumerate(zip(periods_full, params_full)):
    fast, slow, base, anchor, levels = p
    levels_str = ",".join([f"{t:.2f}->{f:.2f}" for t, f in levels])
    print(f"#{i+1} {sd.date()} -> {ed.date()}: f={fast},s={slow},base={base:.0%},anchor={anchor},levels=[{levels_str}]")
