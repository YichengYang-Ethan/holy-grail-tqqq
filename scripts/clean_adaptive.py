"""
合规版全自适应 WF (无 look-ahead)
原则:
1. 固定窗口长度 5y/2y (不做扫描选择)
2. 金字塔参数走 walk-forward 自适应
3. 事件驱动重训用 VIX/drawdown 作为实时可观测信号
4. 报告 ALL 主要变体 (不只 cherry-pick 最优)
"""
import yfinance as yf
import pandas as pd
import numpy as np

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
print(f"  数据: {qqq.index[0].date()} ~ {qqq.index[-1].date()}, {len(qqq)} 天")

# ============================================================
def get_bull(qqq, fast, slow):
    ema_f = qqq.ewm(span=fast, adjust=False).mean()
    ema_s = qqq.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s)
    bull.iloc[:slow] = False
    return bull

def build_position(qqq, bull, ma_period, base_tqqq, levels, anchor):
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
    cal = cagr / abs(mdd) if mdd < 0 else cagr
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

# ============================================================
# 全自适应 WF (固定窗口 5y/2y, 内部所有参数 WF 选)
# ============================================================
SEARCH_GRID = {
    'fast_slow': [(5,100),(5,150),(5,200),(8,150),(8,200),(10,200),(13,200)],
    'base_tqqq': [0.0, 0.50],
    'anchor': ['ma', 'peak'],
    'levels': [
        [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)],
        [(-0.05,0.25),(-0.15,0.50),(-0.25,0.75),(-0.35,1.00)],
        [(-0.05,0.50),(-0.20,1.00)],
    ],
}
# 总组合: 7 × 2 × 2 × 3 = 84 / 训练窗口

def fully_adaptive_wf(qqq, tqqq_full, train_y=5, test_y=2, search_grid=SEARCH_GRID):
    all_returns = []
    chosen = []
    test_periods = []
    start_idx = 252 * train_y
    while start_idx + 252 * test_y <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_y, len(qqq))
        train_idx = qqq.index[train_end - 252*train_y : train_end]
        test_idx = qqq.index[train_end : test_end]

        best_cal = -999; best_p = None
        for (f, s) in search_grid['fast_slow']:
            for base in search_grid['base_tqqq']:
                for anchor in search_grid['anchor']:
                    for levels in search_grid['levels']:
                        bull_t = get_bull(qqq.loc[train_idx], f, s)
                        tpos = build_position(qqq.loc[train_idx], bull_t, s, base, levels, anchor)
                        eq, ret = backtest(tpos, tqqq_full.loc[train_idx])
                        c, m, sh, cal, _, _ = metrics(eq, ret)
                        if cal > best_cal:
                            best_cal = cal
                            best_p = (f, s, base, anchor, levels)

        f, s, base, anchor, levels = best_p
        full_idx = qqq.index[train_end - 252*train_y : test_end]
        bull_t = get_bull(qqq.loc[full_idx], f, s)
        tpos = build_position(qqq.loc[full_idx], bull_t, s, base, levels, anchor)
        tpos_test = tpos.loc[test_idx]
        eq, ret = backtest(tpos_test, tqqq_full.loc[test_idx])
        all_returns.append(ret)
        chosen.append(best_p)
        test_periods.append((test_idx[0], test_idx[-1]))
        start_idx += 252 * test_y
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen, test_periods

# ============================================================
# 事件驱动 WF (VIX/drawdown 触发重训)
# ============================================================
def event_driven_wf(qqq, tqqq_full, vix_a, train_y=5, max_test_y=5,
                     min_test_d=126, vix_thresh=35, dd_thresh=-0.20,
                     search_grid=SEARCH_GRID):
    all_returns = []
    chosen = []
    test_periods = []
    current = 252 * train_y
    while current + min_test_d <= len(qqq):
        train_start = max(0, current - 252*train_y)
        train_idx = qqq.index[train_start:current]

        # 训练
        best_cal = -999; best_p = None
        for (f, s) in search_grid['fast_slow']:
            for base in search_grid['base_tqqq']:
                for anchor in search_grid['anchor']:
                    for levels in search_grid['levels']:
                        bull_t = get_bull(qqq.loc[train_idx], f, s)
                        tpos = build_position(qqq.loc[train_idx], bull_t, s, base, levels, anchor)
                        eq, ret = backtest(tpos, tqqq_full.loc[train_idx])
                        c, m, sh, cal, _, _ = metrics(eq, ret)
                        if cal > best_cal:
                            best_cal = cal
                            best_p = (f, s, base, anchor, levels)

        # 找下个重训点 (事件触发或最长 max_test_y 到期)
        max_end = min(current + 252*max_test_y, len(qqq))
        retrain = max_end
        eq_running = 1.0
        peak_running = 1.0
        for j in range(current + min_test_d, max_end):
            v = vix_a.iloc[j]
            # 简化的回撤代理 (这里用 QQQ 价格代理, 真实应该用账户净值)
            test_slice = qqq.iloc[current:j+1]
            mdd_so_far = (test_slice / test_slice.cummax() - 1).min()
            if (not pd.isna(v) and v > vix_thresh) or mdd_so_far < dd_thresh:
                retrain = j
                break

        test_idx = qqq.index[current:retrain]
        f, s, base, anchor, levels = best_p
        full_idx = qqq.index[train_start:retrain]
        bull_t = get_bull(qqq.loc[full_idx], f, s)
        tpos = build_position(qqq.loc[full_idx], bull_t, s, base, levels, anchor)
        tpos_test = tpos.loc[test_idx]
        eq, ret = backtest(tpos_test, tqqq_full.loc[test_idx])
        all_returns.append(ret)
        chosen.append(best_p)
        test_periods.append((test_idx[0], test_idx[-1]))
        current = retrain

    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen, test_periods

# ============================================================
# 实验 1: 固定 5y/2y 全自适应 WF
# ============================================================
print()
print("=" * 100)
print("【实验 1】固定 5y/2y 全自适应 WF (84 参数组合搜索)")
print("=" * 100)
print("\n搜索空间 (固定，不做事后选择):")
print("  fast×slow: 7 组合 (5/100, 5/150, 5/200, 8/150, 8/200, 10/200, 13/200)")
print("  base_tqqq: [0%, 50%]")
print("  anchor: [MA, Peak]")
print("  levels: 3 种金字塔 (标准/激进/极简)")
print("  → 84 组合 / 训练窗口")
print("\n运行中 ...")

eq_full, ret_full, params_full, periods_full = fully_adaptive_wf(qqq, tqqq_full, 5, 2)
c, m, sh, ca, fv, so = metrics(eq_full, ret_full)
print(f"\n📊 结果:")
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}")
print(f"  $10K → ${fv:,.0f}, $150K → ${fv*15:,.0f}")

print("\n各窗口选出的参数（看是否真的自适应）:")
for i, ((sd, ed), p) in enumerate(zip(periods_full, params_full)):
    f, s, base, anchor, levels = p
    levels_str = "/".join([f"{int(t*100)}→{int(fr*100)}%" for t, fr in levels])
    print(f"  #{i+1} {sd.date()}~{ed.date()}: f={f},s={s},base={int(base*100)}%,anchor={anchor},lvl={levels_str}")

# ============================================================
# 实验 2: 不同窗口长度的全自适应 (报告全部, 不cherry-pick)
# ============================================================
print()
print("=" * 100)
print("【实验 2】不同窗口长度对比 (全部展示, 不选最佳)")
print("=" * 100)
print("注意: 这些数字仅供参考, 选哪个窗口要靠先验, 不能事后选")

window_results = {}
for tr in [3, 5, 7]:
    for te in [1, 2, 3]:
        if 252 * (tr + te) > len(qqq) - 252: continue
        eq, ret, ps, _ = fully_adaptive_wf(qqq, tqqq_full, tr, te)
        c, m, sh, ca, fv, so = metrics(eq, ret)
        window_results[(tr, te)] = (c, m, sh, ca, fv, so)

print(f"\n{'Train×Test':<14} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'终值':>14}")
print("-" * 100)
for (tr, te), (c, m, sh, ca, fv, so) in window_results.items():
    print(f"{tr}y × {te}y       {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}")

# ============================================================
# 实验 3: 事件驱动 WF (实时可观测, 无 look-ahead)
# ============================================================
print()
print("=" * 100)
print("【实验 3】事件驱动 WF (VIX/回撤 实时触发)")
print("=" * 100)
print("规则: VIX > 阈值 或 累计回撤 < 阈值 时立即重训")
print("两者都是 t 时刻可观测信号, 无 look-ahead")

print(f"\n{'VIX阈值':<10} {'DD阈值':<10} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'重训次数':>10}")
print("-" * 90)
event_results = {}
for vt in [25, 30, 35, 40]:
    for dt in [-0.15, -0.20, -0.30]:
        eq, ret, ps, pers = event_driven_wf(qqq, tqqq_full, vix_a, 5, 5, 126, vt, dt)
        c, m, sh, ca, fv, so = metrics(eq, ret)
        event_results[(vt, dt)] = (c, m, sh, ca, fv, so, len(pers))
        print(f"VIX>{vt:<6} DD<{int(dt*100):>3}% {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {len(pers):>10}")

# ============================================================
# 实验 4: 严格 OOS 1999-2010 测试
# ============================================================
print()
print("=" * 100)
print("【实验 4】严格 OOS 1999-2010 测试 (全自适应在崩盘期的真实表现)")
print("=" * 100)
print("用 1999-2010 数据独立跑, 看自适应能否生存 dot-com + 2008")

oos_qqq = qqq.loc[:"2010-02-10"]
oos_tqqq = tqqq_full.loc[:"2010-02-10"]
if len(oos_qqq) > 252 * 7:
    oos_eq, oos_ret, _, _ = fully_adaptive_wf(oos_qqq, oos_tqqq, 5, 2)
    c, m, sh, ca, fv, so = metrics(oos_eq, oos_ret)
    print(f"\n  OOS 1999-2010 全自适应 WF:")
    print(f"    CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Calmar {ca:.3f}")
    print(f"    $150K → ${fv*15:,.0f}")

# ============================================================
# 实验 5: 最终对比 (诚实标注每个的 look-ahead 风险)
# ============================================================
print()
print("=" * 100)
print("【最终对比】所有方法 + 诚实标注 look-ahead 风险")
print("=" * 100)

# 重跑基准 WF QQQ/TQQQ rotation
def wf_qqq_tqqq(qqq, tqqq_full, train_y=5, test_y=2):
    fast_grid = [3, 5, 8, 10, 13]; slow_grid = [50, 100, 150, 200, 250]
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

base_eq, base_ret = wf_qqq_tqqq(qqq, tqqq_full)

print(f"\n{'方法':<46} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'终值':>14}")
print("-" * 110)

# 全自适应固定窗口
c, m, sh, ca, fv, so = metrics(eq_full, ret_full)
print(f"{'全自适应 WF (5y/2y, 84 组合)':<46} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {fv:>14,.0f}")

# 基准 WF QQQ/TQQQ
c, m, sh, ca, fv, so = metrics(base_eq, base_ret)
print(f"{'WF QQQ/TQQQ rotation (5y/2y)':<46} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {fv:>14,.0f}")

# 中位事件驱动
mid_event_key = (35, -0.20)
c, m, sh, ca, fv, so, np_e = event_results[mid_event_key]
print(f"{'事件驱动 WF (VIX>35, DD<-20%)':<46} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {fv:>14,.0f}")

# B&H 同期
test_period = base_ret.index
for name, px in [("TQQQ B&H 同期", tqqq_full.loc[test_period]), ("QQQ B&H 同期", qqq.loc[test_period])]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<46} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {ca:>8.3f} {fv:>14,.0f}")

print()
print("【Look-ahead 风险评估表】")
print("-" * 110)
print(f"{'方法':<46} {'WF 内部':<10} {'窗口选择':<10} {'变体选择':<10} {'网格选择':<10}")
print("-" * 110)
print(f"{'全自适应 WF (5y/2y)':<46} {'✅ 干净':<10} {'✅ 固定':<10} {'✅ 内置':<10} {'⚠️ 先验':<10}")
print(f"{'WF QQQ/TQQQ rotation':<46} {'✅ 干净':<10} {'⚠️ 5y/2y':<10} {'⚠️ 多变体':<10} {'⚠️ 先验':<10}")
print(f"{'事件驱动 WF':<46} {'✅ 干净':<10} {'✅ 自适应':<10} {'✅ 内置':<10} {'⚠️ 先验':<10}")
print(f"{'TQQQ/QQQ Buy & Hold':<46} {'✅ N/A':<10} {'✅ N/A':<10} {'✅ N/A':<10} {'✅ N/A':<10}")
