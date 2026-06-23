# Methodology Summary

## Transitory inflation

```text
epsilon_t = inflation_yoy_t - baseline_t
tinf_n_t = rolling_mean(epsilon_t, n)
```

where `n` is 4, 8, or 12 months.

## Baseline matters

The same CPI data can produce different signals depending on the baseline. Always label the baseline.

## Replication versus live signal

- Full-sample mean is acceptable for ex-post replication, but not for live trading/macro monitoring.
- Shifted rolling/expanding baselines are preferred for live interpretation.

## Decay

The paper estimates persistence using AR(1) logic and rolling windows. Treat decay outputs as model estimates, not certainties. Add warnings when parameters imply invalid or unstable convergence.
