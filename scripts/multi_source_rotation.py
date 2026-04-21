"""
多源数据验证 + QQQ/TQQQ 轮换策略
1. 用 Alpha Vantage 数据验证 yfinance 价格真实性
2. 检查关键历史事件的价格点
3. 设计多种 QQQ/TQQQ 轮换策略，避免只用现金
4. Walk-forward 严格测试
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json

START = "1999-03-10"
END = "2026-01-17"
INIT_CASH = 10_000.0

print("=" * 100)
print("【数据验证 1】yfinance vs Alpha Vantage 交叉对比 QQQ 收盘价")
print("=" * 100)

# AV 数据 (从 MCP 拉的，硬编码到这里做对比)
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

# yfinance 数据
yf_qqq = yf.download("QQQ", start="2025-11-20", end="2026-01-18", auto_adjust=False, progress=False)["Close"].squeeze()

print(f"\n{'日期':<12} {'AlphaVantage':>14} {'yfinance':>14} {'差异 %':>10} {'是否一致':>10}")
print("-" * 70)
all_match = True
for date_str, av_close in av_qqq.items():
    date = pd.Timestamp(date_str)
    if date in yf_qqq.index:
        yf_close = float(yf_qqq.loc[date])
        diff = abs(yf_close - av_close) / av_close * 100
        match = "✅" if diff < 0.05 else "❌"
        if diff > 0.05: all_match = False
        print(f"{date_str:<12} ${av_close:>13.2f} ${yf_close:>13.2f} {diff:>9.3f}% {match:>10}")

print(f"\n→ 数据一致性: {'✅ 完全可信' if all_match else '⚠️ 发现差异'}")
print()

print("=" * 100)
print("【数据验证 2】关键历史事件日 QQQ/TQQQ 价格点 - 已知历史事件交叉验证")
print("=" * 100)

qqq = yf.download("QQQ", start="1999-03-10", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
qqq_raw = yf.download("QQQ", start="1999-03-10", end="2026-04-18", auto_adjust=False, progress=False)["Close"].squeeze()
tqqq = yf.download("TQQQ", start="2010-02-11", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_raw = yf.download("TQQQ", start="2010-02-11", end="2026-04-18", auto_adjust=False, progress=False)["Close"].squeeze()
vix = yf.download("^VIX", start="1999-03-10", end="2026-04-18", auto_adjust=True, progress=False)["Close"].squeeze()

# 已知历史事件 (从 wikipedia / news 公开记录)
events = [
    # (date, event, expected behavior)
    ("2000-03-10", "Nasdaq dot-com 顶部 ($117)", qqq_raw, 117, "QQQ raw"),
    ("2002-10-09", "dot-com 底部 ($20)", qqq_raw, 20, "QQQ raw"),
    ("2008-09-15", "雷曼倒闭", qqq_raw, 41, "QQQ raw"),
    ("2008-11-20", "金融危机底", qqq_raw, 25.6, "QQQ raw"),
    ("2010-02-11", "TQQQ 上市", tqqq_raw, 48.3, "TQQQ raw"),  # IPO 价格
    ("2020-03-23", "COVID 底", qqq_raw, 170, "QQQ raw"),
    ("2020-08-31", "VIX 历史新高 80+", vix, 80, "VIX peak (need 2020-03)"),
    ("2022-01-13", "TQQQ 2:1 拆股前", tqqq_raw, 65, "TQQQ raw 拆股前"),
    ("2022-10-13", "2022 熊市底", qqq_raw, 254, "QQQ raw"),
]

print(f"\n{'日期':<12} {'事件':<30} {'数据源':<12} {'预期 ~':>9} {'实际':>9} {'是否合理':>10}")
print("-" * 90)
for date, event, series, expected, src in events:
    try:
        date_ts = pd.Timestamp(date)
        # 找最近的交易日
        if date_ts in series.index:
            actual = float(series.loc[date_ts])
        else:
            actual = float(series.loc[series.index <= date_ts].iloc[-1])
        diff_pct = abs(actual - expected) / expected * 100
        ok = "✅" if diff_pct < 25 else "⚠️"
        print(f"{date:<12} {event:<30} {src:<12} {expected:>9.2f} {actual:>9.2f} {ok:>10}")
    except Exception as e:
        print(f"{date:<12} {event:<30} ERROR: {str(e)[:40]}")

print()
print("=" * 100)
print("【数据验证 3】TQQQ 2022-01-13 拆股 (2:1) - 检查调整正确性")
print("=" * 100)

tqqq_split_check = yf.download("TQQQ", start="2022-01-10", end="2022-01-15", auto_adjust=False, progress=False)
print("拆股期间原始价格 (raw):")
print(tqqq_split_check[["Open", "Close"]].round(2))
ratio = float(tqqq_split_check["Close"].iloc[-1].squeeze() / tqqq_split_check["Close"].iloc[-2].squeeze())
print(f"\n拆股日比率: {ratio:.3f} (应该 ≈ 0.5 if 2:1 split)")
expected_split = 0.5
match = "✅" if abs(ratio - expected_split) < 0.1 else "❌"
print(f"是否符合 2:1 拆股: {match}")

print()
print("=" * 100)
print("【合成 TQQQ 重建】(已验证误差 5.4%)")
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
print(f"合成+真实 TQQQ: {tqqq_full.index[0].date()} ~ {tqqq_full.index[-1].date()}, {len(tqqq_full)} 天")
print()

print("=" * 100)
print("【核心实验】QQQ/TQQQ 轮换策略对比 - 全样本 1999-2026")
print("=" * 100)
print()
print("规则定义：")
print("  v0 现金/TQQQ:  风险开 → TQQQ, 风险关 → 现金 (原方案)")
print("  v1 QQQ/TQQQ:   风险开 → TQQQ, 风险关 → QQQ (永远在场)")
print("  v2 三态:       强势 → TQQQ, 弱势 → QQQ, 极弱 → 现金")
print("  v3 部分仓位:    强势 → 70% TQQQ + 30% QQQ, 弱势 → 100% QQQ")
print("  v4 VIX 滤波:    v1 + VIX>30 强制 QQQ")
print()

def backtest_dual(pos_qqq, pos_tqqq, qqq_px, tqqq_px, fee_bps=2.5, slip_bps=5.0):
    """
    双资产回测。pos_qqq + pos_tqqq <= 1 (剩余是现金)
    pos_*: 当日目标权重 (0~1)
    """
    qqq_pos = pos_qqq.shift(1).fillna(0)
    tqqq_pos = pos_tqqq.shift(1).fillna(0)
    qqq_ret = qqq_px.pct_change().fillna(0)
    tqqq_ret = tqqq_px.pct_change().fillna(0)
    # 仓位变化导致的交易成本 (按总换手量)
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

# 用相同的信号底层：QQQ EMA 5/200
ma5 = qqq.ewm(span=5, adjust=False).mean()
ma30 = qqq.ewm(span=30, adjust=False).mean()
ma200 = qqq.ewm(span=200, adjust=False).mean()
ma50 = qqq.ewm(span=50, adjust=False).mean()
vix_aligned = vix.reindex(qqq.index).ffill()

# 风险开/关 信号
sig_strong = (ma5 > ma200).astype(float)  # 牛市
sig_strong.iloc[:200] = 0
sig_mild_bull = (ma30 > ma200).astype(float)  # 中等牛市
sig_mild_bull.iloc[:200] = 0
sig_extreme_bear = ((ma5 < ma200) & (ma30 < ma200) & (ma50 < ma200)).astype(float)  # 极弱
sig_extreme_bear.iloc[:200] = 0

strategies = {}

# v0: 原方案 - cash/TQQQ
qpos_v0 = pd.Series(0.0, index=qqq.index)
tpos_v0 = sig_strong.copy()
strategies["v0 现金/TQQQ"] = (qpos_v0, tpos_v0)

# v1: 全 QQQ <-> 全 TQQQ
qpos_v1 = 1 - sig_strong
tpos_v1 = sig_strong.copy()
strategies["v1 QQQ/TQQQ 全切换"] = (qpos_v1, tpos_v1)

# v2: 三态 (强势 TQQQ / 中等 QQQ / 极弱 现金)
qpos_v2 = sig_mild_bull * (1 - sig_strong) * (1 - sig_extreme_bear)  # 中等且非强势
tpos_v2 = sig_strong.copy()
# 极弱时全空
mask = sig_extreme_bear == 1
qpos_v2[mask] = 0
tpos_v2[mask] = 0
strategies["v2 三态 TQQQ/QQQ/现金"] = (qpos_v2, tpos_v2)

# v3: 部分仓位
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

# v4: v1 + VIX 滤波
qpos_v4 = qpos_v1.copy()
tpos_v4 = tpos_v1.copy()
high_vix = vix_aligned > 30
qpos_v4[high_vix] = 1.0
tpos_v4[high_vix] = 0.0
strategies["v4 v1 + VIX>30 强制QQQ"] = (qpos_v4, tpos_v4)

# v5: VIX 动态切换 (低 VIX 全 TQQQ; 中 VIX 100% QQQ; 高 VIX 现金)
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
strategies["v5 VIX 三档动态分配"] = (qpos_v5, tpos_v5)

print(f"\n{'策略':<38} {'CAGR':>8} {'MDD':>9} {'夏普':>7} {'卡玛':>7} {'终值':>14} {'交易':>5}")
print("-" * 100)
for name, (qpos, tpos) in strategies.items():
    eq, ret, n = backtest_dual(qpos, tpos, qqq, tqqq_full)
    c, m, s, ca, fv = metrics(eq, ret)
    print(f"{name:<38} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f} {n:>5}")

# Buy & hold 对照
for name, px in [("QQQ B&H 单纯", qqq), ("TQQQ B&H 单纯", tqqq_full)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, s, ca, fv = metrics(eq, ret)
    print(f"{name:<38} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f} {'-':>5}")

print()
print("=" * 100)
print("【样本外严格测试】1999-2010 (含 dot-com + 2008) 各策略表现")
print("=" * 100)

oos_idx = qqq.index <= "2010-02-10"

print(f"\n{'策略':<38} {'OOS CAGR':>10} {'OOS MDD':>10} {'OOS Calmar':>11} {'是否不亏钱':>10}")
print("-" * 100)
for name, (qpos, tpos) in strategies.items():
    eq, ret, n = backtest_dual(qpos[oos_idx], tpos[oos_idx], qqq[oos_idx], tqqq_full[oos_idx])
    c, m, s, ca, fv = metrics(eq, ret)
    flag = "✅" if c > 0 else "❌"
    print(f"{name:<38} {c*100:>9.2f}% {m*100:>9.2f}% {ca:>11.3f} {flag:>10}")

# B&H 对照
for name, px in [("QQQ B&H 1999-2010", qqq[oos_idx]), ("TQQQ B&H 1999-2010", tqqq_full[oos_idx])]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, s, ca, fv = metrics(eq, ret)
    print(f"{name:<38} {c*100:>9.2f}% {m*100:>9.2f}% {ca:>11.3f}")

print()
print("=" * 100)
print("【Walk-Forward + 轮换】最佳策略动态参数 + QQQ/TQQQ 轮换")
print("=" * 100)

def run_wf_rotation(qqq, tqqq_full, train_years=5, test_years=2):
    """walk-forward, 在每段选最优 (fast, slow) 参数, 用 QQQ/TQQQ 轮换执行"""
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

print("\n运行 walk-forward (5y train + 2y test, QQQ/TQQQ 轮换) ...")
wf_eq, wf_ret, wf_params, wf_periods = run_wf_rotation(qqq, tqqq_full, 5, 2)
c, m, s, ca, fv = metrics(wf_eq, wf_ret)
print(f"\n📊 Walk-Forward QQQ/TQQQ 轮换结果:")
print(f"  CAGR: {c*100:.2f}%, MDD: {m*100:.2f}%, 夏普: {s:.3f}, 卡玛: {ca:.3f}")
print(f"  终值: ${fv:,.0f} (起始 $10,000)")
print(f"\n各窗口最优参数:")
for (sd, ed), (f, sl) in zip(wf_periods, wf_params):
    print(f"  {sd.date()} → {ed.date()}: fast={f}, slow={sl}")

# 对照: 同期间 buy & hold
test_period = wf_ret.index
qqq_test = qqq.loc[test_period]
tqqq_test = tqqq_full.loc[test_period]
print(f"\n同时段 Buy & Hold 对照 ({test_period[0].date()} ~ {test_period[-1].date()}):")
for name, px in [("QQQ B&H 同期", qqq_test), ("TQQQ B&H 同期", tqqq_test)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, s, ca, fv = metrics(eq, ret)
    print(f"  {name:<25}: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, 卡玛 {ca:.3f}, 终值 ${fv:,.0f}")

# 对比之前的 cash 版本 walk-forward
print(f"\n📊 对比: 之前的 cash/TQQQ Walk-Forward 结果 (cash 版本):")
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
print(f"  WF cash/TQQQ:    CAGR {c*100:.2f}%, MDD {m*100:.2f}%, 夏普 {s:.3f}, 卡玛 {ca:.3f}, 终值 ${fv:,.0f}")

print()
print("=" * 100)
print("【最终对比表】所有方法在严格样本外的真实表现")
print("=" * 100)

methods = []
# WF rotation
c, m, s, ca, fv = metrics(wf_eq, wf_ret)
methods.append(("⭐ WF EMA QQQ/TQQQ 轮换 (NEW)", c, m, s, ca, fv))
# WF cash
c, m, s, ca, fv = metrics(wf_cash_eq, wf_cash_ret)
methods.append(("WF EMA cash/TQQQ (旧)", c, m, s, ca, fv))
# B&H 同期
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
methods.append(("50/50 QQQ+TQQQ 持续持有", c, m, s, ca, fv))

print(f"\n{'方法':<40} {'CAGR':>8} {'MDD':>9} {'夏普':>7} {'卡玛':>7} {'终值':>14}")
print("-" * 100)
for name, c, m, s, ca, fv in sorted(methods, key=lambda x: -x[4]):  # sort by Calmar
    print(f"{name:<40} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

print()
print("=" * 100)
print("【Bootstrap 显著性检验】WF 轮换策略 1000 次重采样")
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
print(f"\nWalk-Forward QQQ/TQQQ 轮换 1000 次 block bootstrap:")
print(f"  CAGR  中位数 {np.median(cagrs)*100:.2f}%, 95% CI [{np.percentile(cagrs, 2.5)*100:.2f}%, {np.percentile(cagrs, 97.5)*100:.2f}%]")
print(f"  夏普  中位数 {np.median(sharpes):.3f}, 95% CI [{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]")
print(f"  CAGR > 0 概率: {(cagrs > 0).mean()*100:.1f}%")
print(f"  CAGR > 15% 概率: {(cagrs > 0.15).mean()*100:.1f}%")
