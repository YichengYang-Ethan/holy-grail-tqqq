"""
Target: MDD < 20% while preserving meaningful TQQQ upside exposure.

Candidates tested:
1. Volatility-targeted TQQQ (scale leverage to hit target vol)
2. TQQQ + TLT pair (negative-correlation hedge)
3. Rotation with portfolio-level trailing stop-loss
4. Mixed: 30% Rotation + 70% Permanent Portfolio
5. Mixed: 50% Rotation + 50% TLT
6. TQQQ + PUT hedge (simulated via implied cost)
7. Rotation scaled (30% pos size, rest cash/TLT)
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
tqqq_real = yf.download("TQQQ", start="2010-02-11", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
tlt = yf.download("TLT", start="2002-07-30", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
gld = yf.download("GLD", start="2004-11-18", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
bil = yf.download("BIL", start="2007-05-30", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
spy = yf.download("SPY", start="1999-03-10", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()

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

# Use common date range - TLT starts 2002-07, so use from 2003 onwards
common_start = pd.Timestamp("2003-07-30")
qqq_c = qqq.loc[common_start:]
tqqq_c = tqqq_full.loc[common_start:]
tlt_c = tlt.reindex(qqq_c.index).ffill()
spy_c = spy.loc[common_start:]
gld_c = gld.reindex(qqq_c.index).ffill()  # May have NaN early
bil_c = bil.reindex(qqq_c.index).ffill()

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

# Strategy 1: Vol-targeted TQQQ (scale leverage to hit target vol)
def vol_targeted_tqqq(qqq_d, tqqq_d, target_vol=0.15, lookback=60, bond_d=None):
    """
    Scale position daily using past 60-day TQQQ realized vol, target annualized vol 15%
    Extra cash parked in TLT (if provided)
    """
    tret = tqqq_d.pct_change().fillna(0)
    realized_vol = tret.rolling(lookback).std() * np.sqrt(252)
    # Target leverage on TQQQ to achieve target vol
    leverage = (target_vol / realized_vol).clip(upper=1.0, lower=0.0).fillna(0)
    # Use bull signal as additional gate
    bull = get_bull(qqq_d, FAST, SLOW)
    leverage = leverage.where(bull, 0)

    tpos = leverage
    bond_pos = (1 - tpos) if bond_d is not None else pd.Series(0.0, index=qqq_d.index)

    tpos_l = tpos.shift(1).fillna(0)
    bpos_l = bond_pos.shift(1).fillna(0)
    tqqq_ret = tqqq_d.pct_change().fillna(0)
    bond_ret = bond_d.pct_change().fillna(0) if bond_d is not None else pd.Series(0.0, index=qqq_d.index)

    pos_change = tpos_l.diff().abs().fillna(0) + bpos_l.diff().abs().fillna(0)
    cost = pos_change * 7.5 / 10000
    strat_ret = tpos_l * tqqq_ret + bpos_l * bond_ret - cost
    eq = (1 + strat_ret).cumprod() * INIT
    return eq

# Strategy 2: TQQQ + TLT pair (fixed weights)
def tqqq_tlt_pair(qqq_d, tqqq_d, tlt_d, tqqq_w=0.40, tlt_w=0.60, rebal=63):
    """Fixed 40% TQQQ + 60% TLT, quarterly rebalance"""
    t_ret = tqqq_d.pct_change().fillna(0)
    b_ret = tlt_d.pct_change().fillna(0)
    idx = qqq_d.index
    # Track weights drift between rebals
    t_weight = tqqq_w
    b_weight = tlt_w
    eq_hist = pd.Series(index=idx, dtype=float)
    eq_hist.iloc[0] = INIT
    last_rebal = 0
    for i in range(1, len(idx)):
        t_weight *= (1 + t_ret.iloc[i])
        b_weight *= (1 + b_ret.iloc[i])
        total = t_weight + b_weight
        if i - last_rebal >= rebal:
            # rebal cost
            turnover = abs(t_weight/total - tqqq_w) + abs(b_weight/total - tlt_w)
            total -= total * turnover * 7.5 / 10000
            t_weight = tqqq_w * total
            b_weight = tlt_w * total
            last_rebal = i
        eq_hist.iloc[i] = t_weight + b_weight
    eq_hist = eq_hist / eq_hist.iloc[0] * INIT
    return eq_hist

# Strategy 3: Rotation with portfolio-level trailing stop
def rotation_stop_loss(qqq_d, tqqq_d, stop_pct=0.15):
    """Rotation + force flat when account drawdown from peak exceeds stop_pct"""
    bull = get_bull(qqq_d, FAST, SLOW).astype(float)
    qpos_base = 1 - bull
    tpos_base = bull

    qret = qqq_d.pct_change().fillna(0)
    tret = tqqq_d.pct_change().fillna(0)

    eq = pd.Series(index=qqq_d.index, dtype=float)
    peak = INIT
    in_stopout = False
    current_eq = INIT
    eq.iloc[0] = INIT

    for i in range(1, len(qqq_d.index)):
        if in_stopout:
            # Flat, wait for signal reversal to re-enter
            current_ret = 0
            if bull.iloc[i-1] == 1 and bull.iloc[i-2] == 0:  # new bull signal
                in_stopout = False
        else:
            q = qpos_base.shift(1).iloc[i]
            t = tpos_base.shift(1).iloc[i]
            pc = abs(qpos_base.diff().iloc[i]) + abs(tpos_base.diff().iloc[i]) if i > 0 else 0
            current_ret = q * qret.iloc[i] + t * tret.iloc[i] - pc * 7.5/10000

        current_eq = current_eq * (1 + current_ret)
        if current_eq > peak:
            peak = current_eq

        # Check stop
        if not in_stopout and current_eq / peak - 1 < -stop_pct:
            in_stopout = True

        eq.iloc[i] = current_eq
    return eq

# Strategy 4: 30% Rotation + 70% Permanent Portfolio
def rotation_plus_pp(qqq_d, tqqq_d, spy_d, tlt_d, gld_d, bil_d, rot_w=0.30, rebal=63):
    """30% rotation + 70% Permanent Portfolio (SPY/TLT/GLD/BIL)"""
    bull = get_bull(qqq_d, FAST, SLOW).astype(float)
    qpos = 1 - bull
    tpos = bull
    qret = qqq_d.pct_change().fillna(0)
    tret = tqqq_d.pct_change().fillna(0)
    rot_ret = qpos.shift(1).fillna(0) * qret + tpos.shift(1).fillna(0) * tret
    # turnover cost
    pc = qpos.diff().abs().fillna(0) + tpos.diff().abs().fillna(0)
    rot_ret -= pc * 7.5 / 10000

    # Permanent portfolio sleeve
    pp_ret = 0.25 * spy_d.pct_change().fillna(0) + 0.25 * tlt_d.pct_change().fillna(0) \
           + 0.25 * gld_d.pct_change().fillna(0) + 0.25 * bil_d.pct_change().fillna(0)

    total = rot_w * rot_ret + (1 - rot_w) * pp_ret
    eq = (1 + total).cumprod() * INIT
    return eq

# Strategy 5: Rotation scaled to 50% max exposure
def rotation_scaled(qqq_d, tqqq_d, tlt_d, tqqq_max=0.50):
    """Rotation with max 50% TQQQ + 50% TLT throughout"""
    bull = get_bull(qqq_d, FAST, SLOW).astype(float)
    tpos = bull * tqqq_max
    bpos = (1 - bull) * tqqq_max + (1 - tqqq_max)  # Always hold 50% TLT + bull->TQQQ for the other half
    # simplify: bull -> 50% TQQQ + 50% TLT; bear -> 50% QQQ + 50% TLT (or cash)
    # ACTUALLY simpler: 50% TQQQ when bull, 50% SPY when bear; always 50% TLT
    # Let's do: bull -> 50% TQQQ + 50% TLT; bear -> 50% QQQ + 50% TLT
    qpos = (1 - bull) * tqqq_max  # half in QQQ during bear
    tpos_f = bull * tqqq_max        # half in TQQQ during bull
    bpos = pd.Series(1 - tqqq_max, index=qqq_d.index)  # always 50% TLT

    qret = qqq_d.pct_change().fillna(0)
    tret = tqqq_d.pct_change().fillna(0)
    bret = tlt_d.pct_change().fillna(0)

    strat_ret = qpos.shift(1).fillna(0) * qret + tpos_f.shift(1).fillna(0) * tret + bpos.shift(1).fillna(0) * bret
    # Cost
    pc = qpos.diff().abs().fillna(0) + tpos_f.diff().abs().fillna(0) + bpos.diff().abs().fillna(0)
    strat_ret -= pc * 7.5 / 10000
    eq = (1 + strat_ret).cumprod() * INIT
    return eq

# Strategy 6: Aggressive rotation + daily vol targeting
def dynamic_rotation_vol(qqq_d, tqqq_d, tlt_d, target_vol=0.18, lookback=60):
    """Rotation where position sized to target vol; excess in TLT"""
    bull = get_bull(qqq_d, FAST, SLOW)
    # When bull: target_vol / TQQQ realized vol
    tret = tqqq_d.pct_change().fillna(0)
    tqqq_vol = tret.rolling(lookback).std() * np.sqrt(252)
    # When bear: target_vol / QQQ realized vol
    qret = qqq_d.pct_change().fillna(0)
    qqq_vol = qret.rolling(lookback).std() * np.sqrt(252)

    tqqq_leverage = (target_vol / tqqq_vol).clip(0, 1).fillna(0)
    qqq_leverage = (target_vol / qqq_vol).clip(0, 1).fillna(0)

    tpos = tqqq_leverage.where(bull, 0)
    qpos = qqq_leverage.where(~bull, 0)
    # Cash in TLT
    bond_pos = 1 - tpos - qpos
    bond_pos = bond_pos.clip(lower=0)

    bret = tlt_d.pct_change().fillna(0)
    strat_ret = tpos.shift(1).fillna(0) * tret + qpos.shift(1).fillna(0) * qret + bond_pos.shift(1).fillna(0) * bret
    pc = tpos.diff().abs().fillna(0) + qpos.diff().abs().fillna(0) + bond_pos.diff().abs().fillna(0)
    strat_ret -= pc * 7.5 / 10000
    eq = (1 + strat_ret).cumprod() * INIT
    return eq

# ============================================================
print("\n" + "=" * 108)
print("CANDIDATES: MDD < 20% WITH TQQQ UPSIDE - SINGLE-PATH 2003-2026")
print("=" * 108)
print(f"\n{'Strategy':<50} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Final':>10}")
print("-" * 108)

strategies_eq = {}

# Baseline
from_test = {
    "Rotation (fixed 5/200) - baseline": lambda: compute_base_rotation(qqq_c, tqqq_c),
    "Vol-target TQQQ (15%, TLT)": lambda: vol_targeted_tqqq(qqq_c, tqqq_c, 0.15, 60, tlt_c),
    "Vol-target TQQQ (10%, TLT)": lambda: vol_targeted_tqqq(qqq_c, tqqq_c, 0.10, 60, tlt_c),
    "40/60 TQQQ/TLT quarterly": lambda: tqqq_tlt_pair(qqq_c, tqqq_c, tlt_c, 0.40, 0.60, 63),
    "30/70 TQQQ/TLT quarterly": lambda: tqqq_tlt_pair(qqq_c, tqqq_c, tlt_c, 0.30, 0.70, 63),
    "20/80 TQQQ/TLT quarterly": lambda: tqqq_tlt_pair(qqq_c, tqqq_c, tlt_c, 0.20, 0.80, 63),
    "Rotation + 15% trail stop": lambda: rotation_stop_loss(qqq_c, tqqq_c, 0.15),
    "Rotation + 20% trail stop": lambda: rotation_stop_loss(qqq_c, tqqq_c, 0.20),
    "30% Rotation + 70% PP": lambda: rotation_plus_pp(qqq_c, tqqq_c, spy_c, tlt_c, gld_c, bil_c, 0.30, 63),
    "50% Rotation + 50% PP": lambda: rotation_plus_pp(qqq_c, tqqq_c, spy_c, tlt_c, gld_c, bil_c, 0.50, 63),
    "Half-size rotation + 50% TLT": lambda: rotation_scaled(qqq_c, tqqq_c, tlt_c, 0.50),
    "Dynamic vol-target rotation (18%)": lambda: dynamic_rotation_vol(qqq_c, tqqq_c, tlt_c, 0.18, 60),
    "Dynamic vol-target rotation (12%)": lambda: dynamic_rotation_vol(qqq_c, tqqq_c, tlt_c, 0.12, 60),
}

def compute_base_rotation(qqq_d, tqqq_d):
    bull = get_bull(qqq_d, FAST, SLOW).astype(float)
    qpos = 1 - bull; tpos = bull
    qret = qqq_d.pct_change().fillna(0); tret = tqqq_d.pct_change().fillna(0)
    pc = qpos.diff().abs().fillna(0) + tpos.diff().abs().fillna(0)
    strat_ret = qpos.shift(1).fillna(0) * qret + tpos.shift(1).fillna(0) * tret - pc * 7.5/10000
    return (1 + strat_ret).cumprod() * INIT

# Run all
for name, fn in from_test.items():
    try:
        eq = fn()
        c, m, sh, ca, fv, so = metrics(eq)
        strategies_eq[name] = eq
        flag = "PASS" if m > -0.20 else "WARN"
        print(f"{name:<50} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>9,.0f} {flag}")
    except Exception as e:
        print(f"{name:<50} ERROR: {e}")

# ============================================================
# Add TQQQ B&H + QQQ B&H + Rotation for reference
# ============================================================
print("\n" + "=" * 108)
print("REFERENCE:")
print("-" * 108)
for name, px in [("QQQ Buy & Hold", qqq_c), ("TQQQ Buy & Hold", tqqq_c), ("SPY B&H", spy_c)]:
    eq = (px / px.iloc[0]) * INIT
    c, m, sh, ca, fv, so = metrics(eq)
    print(f"{name:<50} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>9,.0f}")

# ============================================================
# Rank by Calmar, filter to MDD < 20%
# ============================================================
print("\n" + "=" * 108)
print("WINNERS: MDD < 20% RANKED BY CAGR")
print("=" * 108)
print(f"\n{'Strategy':<50} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8}")
print("-" * 108)

valid = []
for name, eq in strategies_eq.items():
    c, m, sh, ca, fv, so = metrics(eq)
    if m > -0.20:
        valid.append((name, c, m, sh, ca, fv))

for name, c, m, sh, ca, fv in sorted(valid, key=lambda x: -x[1]):
    print(f"{name:<50} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f}")

print()
print(f"Total passing MDD < 20%: {len(valid)}")
print("(For reference: Rotation alone has MDD -95%, TQQQ B&H has MDD -99.98%)")
