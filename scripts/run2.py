"""
Precisely match moomoo 47% experiment:
- Screenshot shows EMA5/EMA10/EMA20 displayed -> may not be 5/200 but 5/20
- Could also be same-day close fill (no T+1 lag)
- Try 5/20, 5/10, and several same-day-fill combinations
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "2010-02-11"
END = "2026-01-17"
INIT_CASH = 10_000.0

data = yf.download(["QQQ", "TQQQ"], start=START, end=END, auto_adjust=True, progress=False)
close = data["Close"].dropna()

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def backtest(signal_series, trade_price_series, fast, slow, lag=1, label=""):
    ema_f = ema(signal_series, fast)
    ema_s = ema(signal_series, slow)
    sig = (ema_f > ema_s).astype(int)
    trade_signal = sig.shift(lag).fillna(0)
    valid = trade_signal.index >= trade_signal.index[slow]
    trade_signal = trade_signal[valid]
    px = trade_price_series.loc[trade_signal.index]
    daily_ret = px.pct_change().fillna(0)
    strat_ret = daily_ret * trade_signal
    equity = (1 + strat_ret).cumprod() * INIT_CASH
    n_buy = int(((trade_signal == 1) & (trade_signal.shift(1) == 0)).sum())
    n_sell = int(((trade_signal == 0) & (trade_signal.shift(1) == 1)).sum())
    return equity, strat_ret, n_buy, n_sell

def metrics(equity, daily_ret):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max - 1)
    max_dd = drawdown.min()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0
    return cagr, max_dd, sharpe, calmar, equity.iloc[-1]

qqq = close["QQQ"]
tqqq = close["TQQQ"]

# Multi-combination scan
combos = [
    ("EMA5/200 TQQQ T+1", tqqq, 5, 200, 1),
    ("EMA5/200 TQQQ same-day", tqqq, 5, 200, 0),
    ("EMA5/200 QQQ T+1", qqq, 5, 200, 1),
    ("EMA5/200 QQQ same-day", qqq, 5, 200, 0),
    ("EMA5/20 TQQQ T+1", tqqq, 5, 20, 1),
    ("EMA5/20 TQQQ same-day", tqqq, 5, 20, 0),
    ("EMA5/20 QQQ T+1", qqq, 5, 20, 1),
    ("EMA5/10 TQQQ T+1", tqqq, 5, 10, 1),
    ("EMA10/200 TQQQ T+1", tqqq, 10, 200, 1),
    ("EMA20/200 TQQQ T+1", tqqq, 20, 200, 1),
    ("EMA50/200 TQQQ T+1", tqqq, 50, 200, 1),
]

print(f"{'Strategy':<30} {'CAGR':>10} {'Max DD':>12} {'Sharpe':>8} {'Calmar':>8} {'Final Value':>14} {'Trades':>6}")
print("-" * 100)
for name, sig_src, f, s, lag in combos:
    eq, ret, b, sn = backtest(sig_src, tqqq, f, s, lag, name)
    cagr, dd, sh, ca, fv = metrics(eq, ret)
    n_trades = b + sn
    print(f"{name:<30} {cagr*100:>9.2f}% {dd*100:>11.2f}% {sh:>8.3f} {ca:>8.3f} {fv:>14,.0f} {n_trades:>6}")

print("-" * 100)
print(f"{'moomoo benchmark':<30} {47.08:>9.2f}% {-72.80:>11.2f}% {1.083:>8.3f} {0.647:>8.3f} {425895:>14,.0f} {53:>6}")
