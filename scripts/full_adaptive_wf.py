"""
全自适应 Walk-Forward + 重训频率研究
- 自适应: (fast, slow) + 金字塔参数 (base_tqqq, anchor, levels)
- 频率研究: 训练窗口长度 + 测试窗口长度 + 事件驱动重训
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0

print("[1/5] 数据准备 ...")
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
print(f"  数据范围: {qqq.index[0].date()} ~ {qqq.index[-1].date()}, {len(qqq)} 天")

# ============================================================
def get_bull(qqq, fast, slow):
    ema_f = qqq.ewm(span=fast, adjust=False).mean()
    ema_s = qqq.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s)
    bull.iloc[:slow] = False
    return bull

def build_position(qqq, bull, ma_period, base_tqqq, levels, anchor):
    """生成 TQQQ 仓位序列"""
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
# 全自适应 Walk-Forward
# ============================================================
def fully_adaptive_wf(qqq, tqqq_full, train_y=5, test_y=2, search_grid=None, verbose=False):
    """
    walk-forward 同时优化:
    - (fast, slow) EMA 信号
    - base_tqqq (熊市基础仓位)
    - anchor (ma vs peak)
    - 金字塔阶梯
    """
    if search_grid is None:
        search_grid = {
            'fast_slow': [(3,100),(5,100),(5,150),(5,200),(5,250),(8,100),(8,150),(8,200),(10,150),(10,200),(10,250),(13,200)],
            'base_tqqq': [0.0, 0.25, 0.50, 0.75],
            'anchor': ['ma', 'peak'],
            'levels': [
                [(-0.05,0.25),(-0.10,0.50),(-0.20,0.75),(-0.30,1.00)],  # 早+激进
                [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)],  # 标准
                [(-0.15,0.25),(-0.25,0.50),(-0.35,0.75),(-0.45,1.00)],  # 保守
                [(-0.05,0.50),(-0.20,1.00)],                             # 简化两档
                [(-0.10,0.33),(-0.20,0.66),(-0.30,1.00)],                # 三档
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

        # 网格搜索
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

        # 应用到测试期
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
# 实验 1: 基础全自适应（5y/2y）
# ============================================================
print()
print("=" * 100)
print("【实验 1】全自适应 WF (训练 5y, 测试 2y) — 让所有参数自适应")
print("=" * 100)
print("\n搜索空间:")
print("  - (fast, slow): 12 组合")
print("  - base_tqqq: [0, 0.25, 0.50, 0.75]")
print("  - anchor: [ma, peak]")
print("  - levels: 5 种金字塔配置")
print("  - 总计: 12 × 4 × 2 × 5 = 480 组合 / 每个 train 窗口")
print()

print("运行中（约 3-5 分钟）...")
eq_full, ret_full, params_full, periods_full = fully_adaptive_wf(qqq, tqqq_full, 5, 2, verbose=True)
c, m, sh, ca, fv, so = metrics(eq_full, ret_full)
print(f"\n📊 全自适应 WF 结果:")
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}")
print(f"  终值 ${fv:,.0f}, $150K → ${fv*15:,.0f}")

# ============================================================
# 实验 2: 训练/测试窗口长度扫描
# ============================================================
print()
print("=" * 100)
print("【实验 2】训练 / 测试窗口长度扫描 — 找最优重训频率")
print("=" * 100)

# 用更小的搜索空间加快速度
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

print(f"\n{'Train×Test':<14} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'换参次数':>10} {'终值':>14}")
print("-" * 100)
window_results = {}
for tr, te in window_combos:
    if 252 * tr + 252 * te > len(qqq) - 252:
        continue
    eq, ret, ps, pers = fully_adaptive_wf(qqq, tqqq_full, tr, te, search_grid=small_grid, verbose=False)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    window_results[(tr, te)] = (c, m, sh, ca, fv, so, eq, ret)
    print(f"{tr}y train×{te}y test {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {len(ps):>10} {fv:>14,.0f}")

# 找最优窗口
best_window = max(window_results.items(), key=lambda x: x[1][3])
print(f"\n📊 最优窗口: train={best_window[0][0]}y, test={best_window[0][1]}y → Calmar {best_window[1][3]:.3f}")

# ============================================================
# 实验 3: 事件驱动重训 — 用 VIX / 回撤触发
# ============================================================
print()
print("=" * 100)
print("【实验 3】事件驱动重训 — 当市场制度变化时立即重训")
print("=" * 100)

def event_driven_wf(qqq, tqqq_full, vix_a, train_y=5, max_test_y=5,
                     min_test_d=60, vix_thresh=35, dd_thresh=-0.20,
                     search_grid=None, verbose=False):
    """
    事件驱动重训:
    - 至少 min_test_d 天后才能重训
    - 触发条件 (任一满足):
      * 距上次重训 max_test_y 年到期
      * VIX 突破 vix_thresh (制度转折)
      * 累计回撤 < dd_thresh
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

        # 找下一个重训点
        # 先确定最远 max_test_d 天后的位置
        max_test_d = 252 * max_test_y
        scan_end = min(current_idx + max_test_d, len(qqq))

        # 在 [current_idx + min_test_d, scan_end) 范围内找最早触发条件
        retrain_idx = scan_end  # 默认到期
        equity_proxy = (qqq / qqq.iloc[max(0, current_idx-1)] - 1)  # 简化的回撤代理
        for j in range(current_idx + min_test_d, scan_end):
            v = vix_a.iloc[j]
            # 计算 test 期内累计回撤
            test_slice = qqq.iloc[current_idx:j+1]
            mdd_so_far = (test_slice / test_slice.cummax() - 1).min()
            if (not pd.isna(v) and v > vix_thresh and j > current_idx + min_test_d) or mdd_so_far < dd_thresh:
                retrain_idx = j
                break

        test_end = retrain_idx
        test_idx = qqq.index[current_idx:test_end]

        # 训练: 找最优参数
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

        # 应用
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

print("\n运行事件驱动 WF (VIX>35 或 回撤<-20% 触发, 最长 5y 重训) ...")
event_eq, event_ret, event_params, event_periods = event_driven_wf(
    qqq, tqqq_full, vix_a, train_y=5, max_test_y=5,
    min_test_d=126, vix_thresh=35, dd_thresh=-0.20
)
c, m, sh, ca, fv, so = metrics(event_eq, event_ret)
print(f"\n📊 事件驱动 WF 结果:")
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}")
print(f"  终值 ${fv:,.0f}")
print(f"  共重训 {len(event_periods)} 次")
print(f"\n各重训窗口:")
for i, (sd, ed) in enumerate(event_periods):
    days = (ed - sd).days
    print(f"  #{i+1}: {sd.date()} → {ed.date()} ({days} 天)")

# 多个事件触发阈值扫描
print()
print("--- 不同 VIX 阈值扫描 ---")
print(f"\n{'VIX阈值':<10} {'回撤阈值':<10} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'重训次数':>10}")
print("-" * 80)
for vix_t in [25, 30, 35, 40, 50]:
    for dd_t in [-0.10, -0.20, -0.30]:
        eq, ret, ps, pers = event_driven_wf(qqq, tqqq_full, vix_a, 5, 5, 126, vix_t, dd_t)
        c, m, sh, ca, fv, so = metrics(eq, ret)
        print(f"VIX>{vix_t:<6} DD<{dd_t*100:>4.0f}% {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {len(pers):>10}")

# ============================================================
# 实验 4: 终极对比
# ============================================================
print()
print("=" * 100)
print("【最终对比】所有 WF 策略 + 基准")
print("=" * 100)

# 基准
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

print("\n运行 WF QQQ/TQQQ 轮换基准 ...")
base_eq, base_ret = wf_qqq_tqqq(qqq, tqqq_full)

print(f"\n{'方法':<46} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'终值':>14}")
print("-" * 130)

methods = []
c, m, sh, ca, fv, so = metrics(eq_full, ret_full)
methods.append(("⭐ 全自适应 WF (5y/2y, 480 组合)", c, m, sh, so, ca, fv))
c, m, sh, ca, fv, so = metrics(event_eq, event_ret)
methods.append(("⭐ 事件驱动 WF (VIX>35, DD<-20%)", c, m, sh, so, ca, fv))
c, m, sh, ca, fv, so = metrics(base_eq, base_ret)
methods.append(("基准: WF QQQ/TQQQ 轮换 (5y/2y)", c, m, sh, so, ca, fv))

# Best window from experiment 2
(tr, te), (c, m, sh, ca, fv, so, eq_w, ret_w) = best_window
methods.append((f"⭐ 最优窗口 ({tr}y/{te}y) 全自适应", c, m, sh, so, ca, fv))

# B&H 同期
test_period = base_ret.index
for name, px in [("TQQQ B&H 同期", tqqq_full.loc[test_period]), ("QQQ B&H 同期", qqq.loc[test_period])]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    methods.append((name, c, m, sh, so, ca, fv))

# 排序按 Calmar
methods.sort(key=lambda x: -x[5])
for name, c, m, sh, so, ca, fv in methods:
    print(f"{name:<46} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}")

# 1999-2010 OOS 测试
print()
print("=" * 100)
print("【过拟合检验】最佳全自适应策略 1999-2010 OOS")
print("=" * 100)

# 用 1999-2010 数据完全独立测试 全自适应版
oos_qqq = qqq.loc[:"2010-02-10"]
oos_tqqq = tqqq_full.loc[:"2010-02-10"]
print("\n用 1999-2010 数据独立跑全自适应（完全样本外）...")
if len(oos_qqq) > 252 * 7:
    oos_eq, oos_ret, oos_p, _ = fully_adaptive_wf(oos_qqq, oos_tqqq, 5, 2, search_grid=small_grid)
    c, m, sh, ca, fv, so = metrics(oos_eq, oos_ret)
    print(f"  OOS 1999-2010: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {ca:.3f}")

print()
print("=" * 100)
print("【冠军参数轨迹】查看每个 test 窗口选了什么")
print("=" * 100)
for i, ((sd, ed), p) in enumerate(zip(periods_full, params_full)):
    fast, slow, base, anchor, levels = p
    levels_str = ",".join([f"{t:.2f}→{f:.2f}" for t, f in levels])
    print(f"#{i+1} {sd.date()} → {ed.date()}: f={fast},s={slow},base={base:.0%},anchor={anchor},levels=[{levels_str}]")
