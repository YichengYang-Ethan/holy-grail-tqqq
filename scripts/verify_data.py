"""
数据验证：yfinance vs Stooq vs 多种调整方式
重点验证 TQQQ 拆股调整是否正确
"""
import yfinance as yf
import pandas as pd
import numpy as np
import io
import urllib.request

print("=" * 80)
print("【数据验证 1】yfinance auto_adjust=True (拆股+分红调整)")
print("=" * 80)
tqqq_adj = yf.download("TQQQ", start="2010-02-11", end="2026-01-17", auto_adjust=True, progress=False)
print(f"TQQQ 调整后: 起 {tqqq_adj['Close'].iloc[0].item():.4f}, 终 {tqqq_adj['Close'].iloc[-1].item():.2f}")
print(f"  期间倍数: {(tqqq_adj['Close'].iloc[-1] / tqqq_adj['Close'].iloc[0]).item():.1f}x")

print()
print("=" * 80)
print("【数据验证 2】yfinance auto_adjust=False (raw 价格 + Adj Close 单独列)")
print("=" * 80)
tqqq_raw = yf.download("TQQQ", start="2010-02-11", end="2026-01-17", auto_adjust=False, progress=False)
print(f"TQQQ raw Close: 起 {tqqq_raw['Close'].iloc[0].item():.2f}, 终 {tqqq_raw['Close'].iloc[-1].item():.2f}")
print(f"TQQQ Adj Close: 起 {tqqq_raw['Adj Close'].iloc[0].item():.4f}, 终 {tqqq_raw['Adj Close'].iloc[-1].item():.2f}")

print()
print("=" * 80)
print("【数据验证 3】TQQQ 已知拆股事件")
print("=" * 80)
tk = yf.Ticker("TQQQ")
splits = tk.splits
print("TQQQ 历史拆股:")
print(splits)

print()
print("=" * 80)
print("【数据验证 4】Stooq 数据 (独立第二数据源)")
print("=" * 80)
try:
    url = "https://stooq.com/q/d/l/?s=tqqq.us&i=d"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    csv_data = urllib.request.urlopen(req, timeout=15).read().decode()
    stooq_tqqq = pd.read_csv(io.StringIO(csv_data))
    stooq_tqqq['Date'] = pd.to_datetime(stooq_tqqq['Date'])
    stooq_tqqq = stooq_tqqq.set_index('Date').sort_index()
    print(f"Stooq TQQQ: 起 {stooq_tqqq['Close'].iloc[0]:.2f}, 终 {stooq_tqqq['Close'].iloc[-1]:.2f}")
    print(f"  数据范围: {stooq_tqqq.index[0].date()} ~ {stooq_tqqq.index[-1].date()}")
    print(f"  总记录数: {len(stooq_tqqq)}")

    # 取最近 5 天与 yfinance 对比
    print("\n最近 5 天 Stooq vs yfinance raw:")
    recent_yf = tqqq_raw['Close'].tail(5)
    recent_stooq = stooq_tqqq['Close'].tail(5)
    cmp = pd.DataFrame({'yfinance': recent_yf.values.flatten(), 'Stooq': recent_stooq.values}, index=recent_yf.index)
    print(cmp)
except Exception as e:
    print(f"Stooq fetch failed: {e}")

print()
print("=" * 80)
print("【数据验证 5】QQQ 验证")
print("=" * 80)
qqq_adj = yf.download("QQQ", start="2010-02-11", end="2026-01-17", auto_adjust=True, progress=False)
print(f"QQQ 调整后: 起 {qqq_adj['Close'].iloc[0].item():.2f}, 终 {qqq_adj['Close'].iloc[-1].item():.2f}")
print(f"  期间倍数: {(qqq_adj['Close'].iloc[-1] / qqq_adj['Close'].iloc[0]).item():.2f}x")
qtk = yf.Ticker("QQQ")
print(f"QQQ 拆股: {len(qtk.splits)} 次")
print(qtk.splits)
