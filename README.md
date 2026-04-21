# holy-grail-tqqq

Companion code for [**The Holy Grail of Investing**](https://yichengyang-ethan.github.io/holy-grail) — a 27-year backtest of a simple moving-average rotation between QQQ and TQQQ.

## The strategy

One rule, two ETFs:

- If EMA(5) of QQQ > EMA(200) of QQQ → hold **TQQQ** (3× leveraged Nasdaq)
- Else → hold **QQQ** (unleveraged Nasdaq)

Evaluated at each daily close; trade executes the next close. No discretion, ~3–4 rotations per year.

## Headline results (1999-03-10 → 2026-04-18)

| Strategy                | CAGR   | Max DD   | $10K → final | Sharpe |
| ----------------------- | ------ | -------- | ------------ | ------ |
| **Rotation (this code)**| 15.88% | −95.5%   | **$543,387** | 0.67   |
| QQQ buy-and-hold        | 10.52% | −83.0%   | $150,523     | 0.52   |
| S&P 500 buy-and-hold    | 8.4%   | −55.2%   | $89,000      | 0.49   |
| TQQQ buy-and-hold       | 1.37%  | −99.98%  | $14,439      | 0.22   |

Rotation beats QQQ B&H in **79% of rolling 3-year windows** by an average of 16 percentage points.

## Pre-2010 synthetic TQQQ

TQQQ launched in 2010-02-11. For the 1999–2010 window we reconstruct synthetic TQQQ daily returns from QQQ:

```
synth_tqqq_ret = 3 × QQQ_ret − expense_daily − financing_daily − slippage_daily
```

with year-by-year Fed-funds rates for financing, 0.84% annual expense ratio, and 30 bps annual rebalancing slippage. Calibrated against real TQQQ 2010–2026 for drift correction. See `scripts/synth_tqqq_v2.py`.

## Stress tests

The file `scripts/peer_review_fixes.py` runs seven separate robustness checks after adversarial peer review of the initial result:

1. **Gayed 2016 baseline** — price > SMA(200) → leveraged, else T-bills
2. **Parameter robustness grid** — {SMA, EMA} × {5, 10, 20, 50} × {150, 200, 250}
3. **Deflated Sharpe** under N = 10 / 100 / 1000 effective tests
4. **Start-date sensitivity** — 1999 / 2003 / 2010
5. **Volatility-regime-conditional synthetic error** — how wrong is the pre-2010 reconstruction in high-vol periods?
6. **Peak-to-signal lag** analysis
7. **Tax / slippage** stress scenarios

The file `scripts/fixed_params_cpcv.py` runs **combinatorial purged cross-validation** (45 alternative paths with 21-day embargo) on the fixed EMA(5,200) rotation. Result: the rotation's CPCV distribution sits to the right of QQQ B&H's in every percentile, and is ~3.5× more likely to deliver >15% annualized.

## Repo layout

```
scripts/                    # All backtest code
├── synth_tqqq.py           # 1999–2010 synthetic TQQQ reconstruction (v1)
├── synth_tqqq_v2.py        # Same, with corrected cost model + calibration
├── multi_source_rotation.py # QQQ/TQQQ rotation core logic
├── rotation_vs_cash.py     # Bear-leg variants: QQQ vs cash vs T-bills
├── compare_vs_bh.py        # Buy-and-hold comparisons
├── forward_wf_proper.py    # Causal walk-forward validation
├── fixed_params_cpcv.py    # CPCV on fixed EMA(5,200)
├── peer_review_fixes.py    # 7 robustness checks post-review
├── full_adaptive_wf.py     # Full walk-forward with adaptive params (pre-review)
├── clean_adaptive.py       # Cleaner rewrite of the adaptive walk-forward
├── retail_ma_candidates.py # Candidate scan over retail-implementable MA rules
├── low_dd_strategies.py    # Low-drawdown variants
├── low_dd_with_tqqq.py     # Low-DD with TQQQ exposure
├── qqq_multi_signal.py     # Multi-signal QQQ filter experiments
├── pure_tqqq_pyramid.py    # Pyramid scaling on pure TQQQ
├── sqqq_pyramid.py         # Short-side pyramid via SQQQ
├── wf_pyramid_correct.py   # Corrected pyramid walk-forward
├── verify_data.py          # Data-quality checks on yfinance downloads
├── verify_logic.py         # Logic sanity checks on the rotation rule
├── run.py                  # First-pass runner
└── run2.py                 # Second-pass runner
results/                    # Pickled backtest outputs
├── fixed_cpcv_results.pkl  # CPCV paths (used for fig2_cpcv_distribution.png)
├── forward_wf_results.pkl  # Walk-forward outputs
├── retail_candidates.pkl   # Retail MA candidate scan results
└── cpcv_vs_bh.pkl          # Rotation vs QQQ B&H CPCV comparison
figures/                    # Rendered figures used in the article
├── fig1_equity_curves.png  # 27-year equity curves + 4 crisis zoom panels
├── fig_rolling_dd.png      # Rolling outperformance and drawdown
├── fig2_cpcv_distribution.png # CPCV outcome distribution: rotation vs QQQ B&H
└── fig_interactive.html    # Hover-to-inspect equity curves (embedded in blog)
```

## Running

```bash
pip install -r requirements.txt

# Main result: CPCV on the fixed EMA(5,200) rotation
python scripts/fixed_params_cpcv.py

# Headline backtest + buy-and-hold comparison
python scripts/compare_vs_bh.py

# Full robustness suite from peer review
python scripts/peer_review_fixes.py
```

All scripts download QQQ / TQQQ / BIL data on demand from Yahoo Finance — no API keys needed. Expect a few minutes per script; CPCV takes longest.

## Notes

- Transaction costs modeled at 2.5 bps fee + 5 bps slippage per rotation.
- Pre-2010 synthetic TQQQ is calibrated on 2010–2026 overlap; residual drift is distributed evenly across the synthetic window.
- Drawdown figures use the full 27-year path. The −95.5% drawdown occurred during 2000–2002; the rotation survived it and recovered.
- This is research code, not a production trading system. Past performance does not predict future results, and leveraged ETFs carry real risk.

## License

MIT — see `LICENSE`.
