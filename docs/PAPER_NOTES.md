# Paper Notes

Reference: `references/ssrn-4007466.pdf`

Formula audit reference: `docs/PAPER_FORMULA_REFERENCE.md`

To extract local paper text for methodology review:

```powershell
python scripts/extract_reference_paper.py
```

## Key ideas

- Transitory inflation is treated as persistent deviation from mean-reversion inflation.
- Baseline implementation uses 4-month average deviation as short-term TINF.
- 8-month and 12-month versions are robustness variants.
- Paper estimates persistence through AR(1) and rolling-window rho.
- The decay/convergence logic is useful but should be audited carefully.

## Known ambiguity

The main text describes deviation from mean-reversion inflation. The appendix refers to inflation above a 36-month moving average. This project keeps baseline definitions explicit rather than blending them.

## Upgrade direction

The professional version should test whether TINF predicts future inflation out-of-sample and whether it helps interpret market regimes.
