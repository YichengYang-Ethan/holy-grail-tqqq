"""
代码逻辑严格验证：
1. 用 OPEN 价格做执行 (信号 close[T] → buy/sell open[T+1]) — 无 look-ahead
2. 用 CLOSE 价格做执行 (信号 close[T] → buy/sell close[T+1]) — 无 look-ahead
3. 加入手续费 + 滑点
4. 验证 EMA 计算与 cross 检测
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "2010-02-11"
END = "2026-01-17"
INIT_CASH = 10_000.0
FAST, SLOW = 5, 200

# 拉数据 - 同时拿 Open 和 Close
data = yf.download(["QQQ", "TQQQ"], start=START, end=END, auto_adjust=True, progress=False)
opens = data["Open"].dropna()
closes = data["Close"].dropna()

# 对齐
idx = opens.index.intersection(closes.index)
opens = opens.loc[idx]
closes = closes.loc[idx]

print(f"数据范围: {idx[0].date()} ~ {idx[-1].date()}, {len(idx)} 个交易日")
print()

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def event_driven_backtest(signal_src, exec_price_src, fast, slow,
                          fee_bps=2.5, slip_bps=5.0, exec_lag_days=1,
                          label=""):
    """
    严格 event-driven backtest, 无 look-ahead bias.

    - signal_src: 用什么价格序列算 EMA (close)
    - exec_price_src: 用什么价格成交 (open 或 close)
    - exec_lag_days: 信号 → 成交的延迟 (1 = 次日)
    - fee_bps: 单边手续费 (基点, 1bp = 0.01%)
    - slip_bps: 滑点 (基点)
    """
    ema_f = ema(signal_src, fast)
    ema_s = ema(signal_src, slow)
    sig = (ema_f > ema_s).astype(int)
    sig.iloc[:slow] = 0  # 强制 warm-up 期间无信号

    # 信号 cross 事件（仅在 cross 当天有变化）
    sig_change = sig.diff().fillna(0)  # +1 = 上穿, -1 = 下穿
    cross_dates = sig_change[sig_change != 0].index

    # 模拟逐日 equity
    cash = INIT_CASH
    shares = 0.0
    equity_curve = pd.Series(index=idx, dtype=float)
    last_action = None
    n_buy = 0
    n_sell = 0
    fee_total = 0.0
    slip_total = 0.0

    sig_change_dict = sig_change.to_dict()

    for i, t in enumerate(idx):
        # 检查 exec_lag_days 之前的信号
        target_idx = i - exec_lag_days
        if target_idx >= 0:
            sig_t = idx[target_idx]
            change = sig_change_dict.get(sig_t, 0)
            if change == 1 and shares == 0:  # 上穿，全仓买
                px = exec_price_src.loc[t]
                px_with_slip = float(px) * (1 + slip_bps / 10000)
                # 用全部现金买入
                shares_can_buy = cash / px_with_slip
                cost = shares_can_buy * px_with_slip
                fee = cost * fee_bps / 10000
                slip_cost = shares_can_buy * float(px) * slip_bps / 10000
                fee_total += fee
                slip_total += slip_cost
                cash -= cost + fee
                shares = shares_can_buy
                n_buy += 1
                last_action = ("BUY", t, px_with_slip, shares)
            elif change == -1 and shares > 0:  # 下穿，全仓卖
                px = exec_price_src.loc[t]
                px_with_slip = float(px) * (1 - slip_bps / 10000)
                proceeds = shares * px_with_slip
                fee = proceeds * fee_bps / 10000
                slip_cost = shares * float(px) * slip_bps / 10000
                fee_total += fee
                slip_total += slip_cost
                cash += proceeds - fee
                shares = 0
                n_sell += 1
                last_action = ("SELL", t, px_with_slip, 0)

        # 当日权益 = 现金 + 持仓 × 当日收盘价
        eq = cash + shares * float(closes.loc[t]["TQQQ" if "TQQQ" in str(exec_price_src.name) else "TQQQ"])
        equity_curve.loc[t] = eq

    # 强制最后一天的权益用 TQQQ close
    for i, t in enumerate(idx):
        equity_curve.loc[t] = cash if shares == 0 else cash + shares * float(closes["TQQQ"].loc[t])

    # 重算 equity (修正)
    cash = INIT_CASH
    shares = 0.0
    equity_curve = pd.Series(index=idx, dtype=float)
    fee_total = 0.0
    slip_total = 0.0
    n_buy = 0
    n_sell = 0

    for i, t in enumerate(idx):
        target_idx = i - exec_lag_days
        if target_idx >= 0:
            sig_t = idx[target_idx]
            change = sig_change_dict.get(sig_t, 0)
            if change == 1 and shares == 0:
                px_clean = float(exec_price_src.loc[t])
                px_with_slip = px_clean * (1 + slip_bps / 10000)
                shares_can_buy = cash / (px_with_slip * (1 + fee_bps / 10000))
                cost = shares_can_buy * px_with_slip
                fee = cost * fee_bps / 10000
                fee_total += fee
                slip_total += shares_can_buy * px_clean * slip_bps / 10000
                cash -= cost + fee
                shares = shares_can_buy
                n_buy += 1
            elif change == -1 and shares > 0:
                px_clean = float(exec_price_src.loc[t])
                px_with_slip = px_clean * (1 - slip_bps / 10000)
                proceeds = shares * px_with_slip
                fee = proceeds * fee_bps / 10000
                fee_total += fee
                slip_total += shares * px_clean * slip_bps / 10000
                cash += proceeds - fee
                shares = 0
                n_sell += 1

        eq = cash + shares * float(closes["TQQQ"].loc[t])
        equity_curve.loc[t] = eq

    # 计算指标
    daily_ret = equity_curve.pct_change().fillna(0)
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_dd = drawdown.min()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0
    return {
        "label": label,
        "CAGR": cagr,
        "MDD": max_dd,
        "Sharpe": sharpe,
        "Calmar": calmar,
        "FinalValue": equity_curve.iloc[-1],
        "NTrades": n_buy + n_sell,
        "Fees": fee_total,
        "Slippage": slip_total,
    }


print("=" * 100)
print("严格 event-driven 回测（无 look-ahead）+ 手续费 2.5bp + 滑点 5bp")
print("=" * 100)

scenarios = [
    # (label, signal_src, exec_price, exec_lag)
    ("EMA5/200 TQQQ-sig → fill open[T+1]",  closes["TQQQ"], opens["TQQQ"], 1),
    ("EMA5/200 TQQQ-sig → fill close[T+1]", closes["TQQQ"], closes["TQQQ"], 1),
    ("EMA5/200 QQQ-sig  → fill open[T+1]",  closes["QQQ"],  opens["TQQQ"], 1),
    ("EMA5/200 QQQ-sig  → fill close[T+1]", closes["QQQ"],  closes["TQQQ"], 1),
    # 不诚实的对照组：信号 = close[T], 也假装在 close[T] 当日成交
    ("[作弊] EMA5/200 TQQQ same-day close",  closes["TQQQ"], closes["TQQQ"], 0),
]

print(f"\n{'策略':<48} {'CAGR':>9} {'MDD':>9} {'夏普':>8} {'卡玛':>8} {'终值':>14} {'交易':>5}")
print("-" * 110)
results = []
for label, sig_src, exec_px, lag in scenarios:
    r = event_driven_backtest(sig_src, exec_px, FAST, SLOW, exec_lag_days=lag, label=label)
    results.append(r)
    print(f"{label:<48} {r['CAGR']*100:>8.2f}% {r['MDD']*100:>8.2f}% {r['Sharpe']:>8.3f} {r['Calmar']:>8.3f} {r['FinalValue']:>14,.0f} {r['NTrades']:>5}")

# Buy & hold 对照
def bh_metrics(price_series, label):
    eq = (price_series / price_series.iloc[0]) * INIT_CASH
    daily_ret = eq.pct_change().fillna(0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    max_dd = (eq / eq.cummax() - 1).min()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))
    calmar = cagr / abs(max_dd)
    return label, cagr, max_dd, sharpe, calmar, eq.iloc[-1]

print()
print("Buy & Hold 对照:")
print("-" * 110)
for label, c, m, s, ca, f in [bh_metrics(closes["TQQQ"], "TQQQ B&H"), bh_metrics(closes["QQQ"], "QQQ B&H")]:
    print(f"{label:<48} {c*100:>8.2f}% {m*100:>8.2f}% {s:>8.3f} {ca:>8.3f} {f:>14,.0f}     -")

print()
print("moomoo 截图基准:")
print("-" * 110)
print(f"{'moomoo (新建策略1)':<48} {47.08:>8.2f}% {-72.80:>8.2f}% {1.083:>8.3f} {0.647:>8.3f} {425895:>14,.0f}    53")

print()
print("=" * 100)
print("【手续费/滑点详情】")
print("=" * 100)
for r in results:
    print(f"{r['label']:<48} 手续费 ${r['Fees']:>9,.0f}  滑点 ${r['Slippage']:>9,.0f}")
