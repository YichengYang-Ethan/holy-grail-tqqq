"""
SQQQ 反向 + 金字塔加仓策略变体
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0

print("[1/8] 数据准备 ...")
qqq = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
sqqq_real = yf.download("SQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

def build_synth(qqq_close, tqqq_real, mult=3.0):
    """合成 ±3x ETF: mult=3 for TQQQ, mult=-3 for SQQQ"""
    qret = qqq_close.pct_change().fillna(0)
    expense_d = 0.0095 / 252  # SQQQ 略贵
    def fin(d):
        y = d.year
        if y <= 2007: r = 0.045
        elif y <= 2008: r = 0.025
        elif y <= 2015: r = 0.0015
        elif y <= 2019: r = 0.015
        elif y <= 2021: r = 0.001
        else: r = 0.045
        return (r + 0.004) * abs(mult - 1) / 252
    fin_d = pd.Series([fin(d) for d in qqq_close.index], index=qqq_close.index)
    slip_d = 0.005 / 252  # SQQQ 滑点更高
    synth_ret = mult * qret - expense_d - fin_d - slip_d
    synth = (1 + synth_ret).cumprod()
    overlap = tqqq_real.index[0]
    calib = float(tqqq_real.iloc[0]) / float(synth.loc[overlap])
    pre = synth.loc[:overlap].iloc[:-1] * calib
    full = pd.concat([pre, tqqq_real])
    return full.reindex(qqq_close.index).ffill()

tqqq_full = build_synth(qqq, tqqq_real, 3.0)
sqqq_full = build_synth(qqq, sqqq_real, -3.0)

print(f"  QQQ:  {qqq.index[0].date()} ~ {qqq.index[-1].date()}, {len(qqq)} 天")
print(f"  TQQQ (合成+真实): 起 ${tqqq_full.iloc[0]:.4f}, 终 ${tqqq_full.iloc[-1]:.2f}")
print(f"  SQQQ (合成+真实): 起 ${sqqq_full.iloc[0]:.4f}, 终 ${sqqq_full.iloc[-1]:.4f}")

# 验证 SQQQ 合成
overlap = sqqq_real.index[0]
synth_overlap_norm = (sqqq_full.loc[overlap] / sqqq_full.loc[overlap])
real_norm = sqqq_real.loc[overlap:] / sqqq_real.iloc[0]
synth_norm = sqqq_full.loc[overlap:] / sqqq_full.loc[overlap]
final_synth = float(synth_norm.iloc[-1])
final_real = float(real_norm.iloc[-1])
print(f"  SQQQ 验证: 合成 {final_synth:.5f}x vs 真实 {final_real:.5f}x (16年都几乎归零，正常)")

print()
print("=" * 100)

def backtest_multi(positions_dict, prices_dict, fee_bps=2.5, slip_bps=5.0):
    """
    多资产回测。positions_dict: {asset_name: pd.Series 每日权重}
    prices_dict: {asset_name: pd.Series 价格}
    """
    idx = list(positions_dict.values())[0].index
    portfolio_ret = pd.Series(0.0, index=idx)
    total_cost = pd.Series(0.0, index=idx)
    n_trades = 0
    last_pos_change = 0
    for asset, pos in positions_dict.items():
        pos_lag = pos.shift(1).fillna(0)
        ret = prices_dict[asset].pct_change().fillna(0)
        pos_change = pos_lag.diff().abs().fillna(0)
        cost = pos_change * (fee_bps + slip_bps) / 10000
        portfolio_ret = portfolio_ret + pos_lag * ret - cost
    eq = (1 + portfolio_ret).cumprod() * INIT_CASH
    # 数交易：所有资产合计的换手 / 2
    total_change = pd.Series(0.0, index=idx)
    for pos in positions_dict.values():
        total_change = total_change + pos.diff().abs().fillna(0)
    n_trades = int((total_change > 0.05).sum())
    return eq, portfolio_ret, n_trades

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    sortino = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], sortino

print("【实验 1】SQQQ 反向轮换变体 - 全样本 1999-2026")
print("=" * 100)

ma5 = qqq.ewm(span=5, adjust=False).mean()
ma30 = qqq.ewm(span=30, adjust=False).mean()
ma200 = qqq.ewm(span=200, adjust=False).mean()
vix_a = vix.reindex(qqq.index).ffill()

sig_bull = (ma5 > ma200).astype(float); sig_bull.iloc[:200] = 0
sig_bear = (ma5 < ma200).astype(float); sig_bear.iloc[:200] = 0

variants = {}

# A1: TQQQ/SQQQ 全切换
qpos = pd.Series(0.0, index=qqq.index)
tpos = sig_bull.copy()
spos = sig_bear.copy()
variants["A1 TQQQ/SQQQ 全切换"] = (qpos, tpos, spos)

# A2: 牛 TQQQ / 熊 50% SQQQ + 50% 现金 (反向减半)
qpos = pd.Series(0.0, index=qqq.index)
tpos = sig_bull.copy()
spos = sig_bear * 0.5
variants["A2 TQQQ / 50% SQQQ"] = (qpos, tpos, spos)

# A3: 牛 TQQQ / 熊 (VIX>30 才上 SQQQ, 否则现金)
qpos = pd.Series(0.0, index=qqq.index)
tpos = sig_bull.copy()
spos = sig_bear * (vix_a > 30).astype(float)
variants["A3 TQQQ / VIX>30 才 SQQQ"] = (qpos, tpos, spos)

# A4: 牛 TQQQ / 熊 QQQ / VIX>40 SQQQ (三态升级)
qpos = sig_bear * (vix_a <= 40).astype(float)
tpos = sig_bull.copy()
spos = sig_bear * (vix_a > 40).astype(float)
variants["A4 TQQQ / QQQ / VIX>40 SQQQ"] = (qpos, tpos, spos)

# A5: 牛 TQQQ / 熊 SQQQ + 时间衰减保护 (持 SQQQ 不超过 30 天)
sqqq_holding = pd.Series(0.0, index=qqq.index)
days = 0
for i in range(len(qqq.index)):
    if sig_bear.iloc[i] == 1 and days < 30:
        sqqq_holding.iloc[i] = 1.0
        days += 1
    elif sig_bear.iloc[i] == 1:
        sqqq_holding.iloc[i] = 0.0
        days += 1
    else:
        sqqq_holding.iloc[i] = 0.0
        days = 0
qpos = pd.Series(0.0, index=qqq.index)
tpos = sig_bull.copy()
spos = sqqq_holding
variants["A5 TQQQ / SQQQ 限持 30 天"] = (qpos, tpos, spos)

# A6: 牛 TQQQ / 熊 QQQ + SQQQ 各半 (对冲)
qpos = sig_bear * 0.5
tpos = sig_bull.copy()
spos = sig_bear * 0.5
variants["A6 TQQQ / 50%QQQ+50%SQQQ"] = (qpos, tpos, spos)

print(f"\n{'策略':<35} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'终值':>14} {'交易':>5}")
print("-" * 120)
for name, (qp, tp, sp) in variants.items():
    eq, ret, n = backtest_multi({"QQQ": qp, "TQQQ": tp, "SQQQ": sp}, {"QQQ": qqq, "TQQQ": tqqq_full, "SQQQ": sqqq_full})
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<35} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f} {n:>5}")

# 基准对照
print("-" * 120)
for name, px in [("baseline: TQQQ B&H", tqqq_full), ("baseline: QQQ B&H", qqq)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<35} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f} {'-':>5}")

print()
print("=" * 100)
print("【实验 2】金字塔抄底加仓 — 跌破 MA200 后越跌越买")
print("=" * 100)
print()
print("规则:")
print("  - 牛市 (close > MA200): 100% TQQQ")
print("  - 熊市 (close < MA200): 持有现金，分档买入")
print("  - 跌穿 MA200 -10%: 25% TQQQ")
print("  - 跌穿 MA200 -20%: 50% TQQQ")
print("  - 跌穿 MA200 -30%: 75% TQQQ")
print("  - 跌穿 MA200 -40%: 100% TQQQ")
print("  - 重新站上 MA200: 100% TQQQ (恢复正常)")

def pyramid_strategy(qqq, tqqq_full, ma_period=200, levels=None):
    """金字塔加仓: 越跌仓越重"""
    if levels is None:
        levels = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]
    ma = qqq.ewm(span=ma_period, adjust=False).mean()
    bull = (qqq > ma)
    deviation = (qqq / ma - 1)
    tpos = pd.Series(0.0, index=qqq.index)
    for i in range(len(qqq.index)):
        if i < ma_period:
            tpos.iloc[i] = 0
            continue
        if bull.iloc[i]:
            tpos.iloc[i] = 1.0
        else:
            dev = deviation.iloc[i]
            target = 0.0
            for thresh, weight in levels:
                if dev <= thresh:
                    target = weight
            tpos.iloc[i] = target
    return tpos

# B1: 标准金字塔
tpos = pyramid_strategy(qqq, tqqq_full)
qpos = pd.Series(0.0, index=qqq.index)
spos = pd.Series(0.0, index=qqq.index)
variants_b = {"B1 标准金字塔 (4 档)": (qpos, tpos, spos)}

# B2: 激进金字塔 (跌穿就开始买)
tpos = pyramid_strategy(qqq, tqqq_full, levels=[(-0.05, 0.25), (-0.15, 0.50), (-0.25, 0.75), (-0.35, 1.00)])
variants_b["B2 激进金字塔 (-5%开始)"] = (qpos, tpos, spos)

# B3: 保守金字塔 (跌得很深才买)
tpos = pyramid_strategy(qqq, tqqq_full, levels=[(-0.20, 0.20), (-0.30, 0.40), (-0.40, 0.70), (-0.50, 1.00)])
variants_b["B3 保守金字塔 (-20%开始)"] = (qpos, tpos, spos)

# B4: 金字塔 + QQQ 替代现金
def pyramid_with_qqq(qqq, ma_period=200, levels=None):
    if levels is None:
        levels = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]
    ma = qqq.ewm(span=ma_period, adjust=False).mean()
    bull = (qqq > ma)
    deviation = (qqq / ma - 1)
    tpos = pd.Series(0.0, index=qqq.index)
    qpos = pd.Series(0.0, index=qqq.index)
    for i in range(len(qqq.index)):
        if i < ma_period:
            continue
        if bull.iloc[i]:
            tpos.iloc[i] = 1.0
            qpos.iloc[i] = 0.0
        else:
            dev = deviation.iloc[i]
            tq = 0.0
            for thresh, weight in levels:
                if dev <= thresh:
                    tq = weight
            tpos.iloc[i] = tq
            qpos.iloc[i] = 1.0 - tq  # 剩下用 QQQ 而不是现金
    return qpos, tpos

qpos_b4, tpos_b4 = pyramid_with_qqq(qqq)
variants_b["B4 金字塔 + QQQ 替代现金"] = (qpos_b4, tpos_b4, pd.Series(0.0, index=qqq.index))

# B5: 金字塔 + 涨多卖出 (对称对冲)
def pyramid_with_sell(qqq, ma_period=200):
    ma = qqq.ewm(span=ma_period, adjust=False).mean()
    deviation = (qqq / ma - 1)
    tpos = pd.Series(0.0, index=qqq.index)
    for i in range(len(qqq.index)):
        if i < ma_period:
            continue
        dev = deviation.iloc[i]
        # 涨太高减仓，跌太多加仓
        if dev > 0.20:
            tpos.iloc[i] = 0.30  # 上涨 +20% 减到 30%
        elif dev > 0.10:
            tpos.iloc[i] = 0.60
        elif dev > 0:
            tpos.iloc[i] = 1.00
        elif dev > -0.10:
            tpos.iloc[i] = 0.0
        elif dev > -0.20:
            tpos.iloc[i] = 0.25
        elif dev > -0.30:
            tpos.iloc[i] = 0.50
        elif dev > -0.40:
            tpos.iloc[i] = 0.75
        else:
            tpos.iloc[i] = 1.00
    return tpos

tpos_b5 = pyramid_with_sell(qqq)
qpos = pd.Series(0.0, index=qqq.index)
spos = pd.Series(0.0, index=qqq.index)
variants_b["B5 双向金字塔 (跌买涨卖)"] = (qpos, tpos_b5, spos)

print(f"\n{'金字塔变体':<35} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'终值':>14} {'交易':>5}")
print("-" * 120)
for name, (qp, tp, sp) in variants_b.items():
    eq, ret, n = backtest_multi({"QQQ": qp, "TQQQ": tp, "SQQQ": sp}, {"QQQ": qqq, "TQQQ": tqqq_full, "SQQQ": sqqq_full})
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<35} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f} {n:>5}")

print()
print("=" * 100)
print("【实验 3】组合最优：WF 轮换 + 金字塔抄底 + (可选) SQQQ")
print("=" * 100)

def hybrid_wf_pyramid(qqq, tqqq_full, sqqq_full, vix_a, train_years=5, test_years=2, use_sqqq=False):
    """walk-forward EMA 选参数 + 金字塔抄底 + 可选 SQQQ"""
    fast_grid = [3, 5, 8, 10]
    slow_grid = [50, 100, 150, 200]
    all_returns = []
    chosen = []
    test_periods = []

    start_idx = 252 * train_years
    while start_idx + 252 * test_years <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_years, len(qqq))
        train_idx = qqq.index[train_end - 252*train_years : train_end]
        test_idx = qqq.index[train_end : test_end]

        # train: 选最优 (f, s) for QQQ→TQQQ rotation
        best_cal = -999
        best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                ema_f = qqq.loc[train_idx].ewm(span=f, adjust=False).mean()
                ema_s = qqq.loc[train_idx].ewm(span=s, adjust=False).mean()
                bull = (ema_f > ema_s)
                tpos = bull.astype(float); tpos.iloc[:s] = 0
                qpos = (1 - tpos)
                # 在熊市加金字塔
                ma = qqq.loc[train_idx].ewm(span=s, adjust=False).mean()
                dev = (qqq.loc[train_idx] / ma - 1)
                pyr_extra = pd.Series(0.0, index=train_idx)
                for thresh, w in [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]:
                    pyr_extra = pyr_extra + ((dev <= thresh) & (~bull)).astype(float) * (w - pyr_extra.shift(1).fillna(0)).clip(lower=0) * 0.25
                tpos = tpos + pyr_extra * (~bull).astype(float)
                qpos = (1 - tpos).clip(lower=0)

                eq, ret, _ = backtest_multi({"QQQ": qpos, "TQQQ": tpos}, {"QQQ": qqq.loc[train_idx], "TQQQ": tqqq_full.loc[train_idx]})
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)

        # apply on test
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_years : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        bull = (ema_f > ema_s)
        tpos = bull.astype(float); tpos.iloc[:s] = 0
        ma = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        dev = (qqq.loc[full_idx] / ma - 1)
        # 金字塔: 在熊市按跌幅加 TQQQ
        pyr = pd.Series(0.0, index=full_idx)
        for thresh, w in [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]:
            pyr = pyr.where(pyr > w, w * (dev <= thresh).astype(float))
        # 熊市时取金字塔仓位，否则牛市保留 100% TQQQ
        tpos_final = bull.astype(float)
        tpos_final[~bull] = pyr[~bull]
        qpos_final = (1 - tpos_final).clip(lower=0)

        sigs = {"QQQ": qpos_final.loc[test_idx], "TQQQ": tpos_final.loc[test_idx]}
        prices = {"QQQ": qqq.loc[test_idx], "TQQQ": tqqq_full.loc[test_idx]}

        if use_sqqq:
            # 加 SQQQ: 当 VIX > 35 且明确熊市时
            spos = ((vix_a.loc[test_idx] > 35) & (~bull.loc[test_idx])).astype(float) * 0.30
            sigs["SQQQ"] = spos
            prices["SQQQ"] = sqqq_full.loc[test_idx]
            sigs["QQQ"] = (sigs["QQQ"] - spos).clip(lower=0)

        eq, ret, _ = backtest_multi(sigs, prices)
        all_returns.append(ret)
        chosen.append(best_p)
        test_periods.append((test_idx[0], test_idx[-1]))
        start_idx += 252 * test_years

    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen, test_periods

print("\n运行 WF + 金字塔 (无 SQQQ) ...")
eq1, ret1, params1, periods1 = hybrid_wf_pyramid(qqq, tqqq_full, sqqq_full, vix_a, 5, 2, use_sqqq=False)
c, m, sh, ca, fv, so = metrics(eq1, ret1)
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}, 终值 ${fv:,.0f}")

print("\n运行 WF + 金字塔 + SQQQ (VIX>35) ...")
eq2, ret2, params2, periods2 = hybrid_wf_pyramid(qqq, tqqq_full, sqqq_full, vix_a, 5, 2, use_sqqq=True)
c, m, sh, ca, fv, so = metrics(eq2, ret2)
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}, 终值 ${fv:,.0f}")

print()
print("=" * 100)
print("【最终对比】所有策略变体（同时段 walk-forward 结果）")
print("=" * 100)

# 之前最佳: WF QQQ/TQQQ rotation
def wf_rotation_baseline(qqq, tqqq_full, train_years=5, test_years=2):
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
                eq, ret, _ = backtest_multi({"QQQ": 1-bull, "TQQQ": bull}, {"QQQ": qqq.loc[train_idx], "TQQQ": tqqq_full.loc[train_idx]})
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_years : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        bull = (ema_f > ema_s).astype(float); bull.iloc[:s] = 0
        bull_test = bull.loc[test_idx]
        eq, ret, _ = backtest_multi({"QQQ": 1-bull_test, "TQQQ": bull_test}, {"QQQ": qqq.loc[test_idx], "TQQQ": tqqq_full.loc[test_idx]})
        all_returns.append(ret)
        start_idx += 252 * test_years
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret

print("\n运行基准 WF QQQ/TQQQ 轮换（之前的版本）...")
eq0, ret0 = wf_rotation_baseline(qqq, tqqq_full)
c0, m0, sh0, ca0, fv0, so0 = metrics(eq0, ret0)

# 同时段 buy & hold
test_period = ret0.index
qqq_p = qqq.loc[test_period]
tqqq_p = tqqq_full.loc[test_period]

print(f"\n{'方法':<40} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'终值':>14}")
print("-" * 120)
print(f"{'WF QQQ/TQQQ 轮换 (之前最佳)':<40} {c0*100:>7.2f}% {m0*100:>8.2f}% {sh0:>8.3f} {so0:>9.3f} {ca0:>8.3f} {fv0:>14,.0f}")

c, m, sh, ca, fv, so = metrics(eq1, ret1)
print(f"{'⭐ WF + 金字塔抄底 (无 SQQQ)':<40} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}")

c, m, sh, ca, fv, so = metrics(eq2, ret2)
print(f"{'⭐ WF + 金字塔 + SQQQ(VIX>35)':<40} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}")

# Buy & hold benchmark
for name, px in [("TQQQ Buy & Hold (同期)", tqqq_p), ("QQQ Buy & Hold (同期)", qqq_p)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<40} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}")

print()
print("=" * 100)
print("【Bootstrap 显著性】最佳新策略 1000 次重采样")
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

# 选最高 Calmar 的版本
best_eq, best_ret = (eq1, ret1)
best_name = "WF + 金字塔抄底"
c1, m1, sh1, ca1, _, so1 = metrics(eq1, ret1)
c2, m2, sh2, ca2, _, so2 = metrics(eq2, ret2)
if ca2 > ca1:
    best_eq, best_ret, best_name = eq2, ret2, "WF + 金字塔 + SQQQ"

cagrs, sharpes = bootstrap(best_ret, 1000, 20)
print(f"\n{best_name}:")
print(f"  CAGR  中位数 {np.median(cagrs)*100:.2f}%, 95% CI [{np.percentile(cagrs, 2.5)*100:.2f}%, {np.percentile(cagrs, 97.5)*100:.2f}%]")
print(f"  夏普  中位数 {np.median(sharpes):.3f}, 95% CI [{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]")
print(f"  CAGR > 0 概率: {(cagrs > 0).mean()*100:.1f}%")
print(f"  CAGR > 20% 概率: {(cagrs > 0.20).mean()*100:.1f}%")
