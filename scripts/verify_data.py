"""
Data verification: yfinance vs Stooq vs various adjustment methods
Focus: verify TQQQ split adjustment correctness
"""
import yfinance as yf
import pandas as pd
import numpy as np
import io
import urllib.request

print("=" * 80)
print("[Data Verification 1] yfinance auto_adjust=True (split + dividend adjusted)")
print("=" * 80)
tqqq_adj = yf.download("TQQQ", start="2010-02-11", end="2026-01-17", auto_adjust=True, progress=False)
print(f"TQQQ adjusted: start {tqqq_adj['Close'].iloc[0].item():.4f}, end {tqqq_adj['Close'].iloc[-1].item():.2f}")
print(f"  Period multiple: {(tqqq_adj['Close'].iloc[-1] / tqqq_adj['Close'].iloc[0]).item():.1f}x")

print()
print("=" * 80)
print("[Data Verification 2] yfinance auto_adjust=False (raw prices + separate Adj Close column)")
print("=" * 80)
tqqq_raw = yf.download("TQQQ", start="2010-02-11", end="2026-01-17", auto_adjust=False, progress=False)
print(f"TQQQ raw Close: start {tqqq_raw['Close'].iloc[0].item():.2f}, end {tqqq_raw['Close'].iloc[-1].item():.2f}")
print(f"TQQQ Adj Close: start {tqqq_raw['Adj Close'].iloc[0].item():.4f}, end {tqqq_raw['Adj Close'].iloc[-1].item():.2f}")

print()
print("=" * 80)
print("[Data Verification 3] Known TQQQ split events")
print("=" * 80)
tk = yf.Ticker("TQQQ")
splits = tk.splits
print("TQQQ split history:")
print(splits)

print()
print("=" * 80)
print("[Data Verification 4] Stooq data (independent second data source)")
print("=" * 80)
try:
    url = "https://stooq.com/q/d/l/?s=tqqq.us&i=d"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    csv_data = urllib.request.urlopen(req, timeout=15).read().decode()
    stooq_tqqq = pd.read_csv(io.StringIO(csv_data))
    stooq_tqqq['Date'] = pd.to_datetime(stooq_tqqq['Date'])
    stooq_tqqq = stooq_tqqq.set_index('Date').sort_index()
    print(f"Stooq TQQQ: start {stooq_tqqq['Close'].iloc[0]:.2f}, end {stooq_tqqq['Close'].iloc[-1]:.2f}")
    print(f"  Data range: {stooq_tqqq.index[0].date()} ~ {stooq_tqqq.index[-1].date()}")
    print(f"  Total records: {len(stooq_tqqq)}")

    # Take last 5 days and compare with yfinance
    print("\nLast 5 days Stooq vs yfinance raw:")
    recent_yf = tqqq_raw['Close'].tail(5)
    recent_stooq = stooq_tqqq['Close'].tail(5)
    cmp = pd.DataFrame({'yfinance': recent_yf.values.flatten(), 'Stooq': recent_stooq.values}, index=recent_yf.index)
    print(cmp)
except Exception as e:
    print(f"Stooq fetch failed: {e}")

print()
print("=" * 80)
print("[Data Verification 5] QQQ verification")
print("=" * 80)
qqq_adj = yf.download("QQQ", start="2010-02-11", end="2026-01-17", auto_adjust=True, progress=False)
print(f"QQQ adjusted: start {qqq_adj['Close'].iloc[0].item():.2f}, end {qqq_adj['Close'].iloc[-1].item():.2f}")
print(f"  Period multiple: {(qqq_adj['Close'].iloc[-1] / qqq_adj['Close'].iloc[0]).item():.2f}x")
qtk = yf.Ticker("QQQ")
print(f"QQQ splits: {len(qtk.splits)} times")
print(qtk.splits)
