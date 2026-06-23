# Validation Protocol

## Code validation

- `pytest` passes
- `python -m compileall src app scripts` passes
- no hardcoded absolute paths
- no API keys in committed files

## Research validation

Check that:

- inflation units are percentage points
- TINF windows are 4/8/12 months
- baseline choice is disclosed
- live-safe mode uses shifted baselines
- paper replication mode and live mode are separated
- AR parameters are extracted by name
- rolling-window dates align with the end of the estimation window
- invalid decay cases are handled

## Statistical validation

Add/maintain checks for:

- summary statistics
- correlation matrix
- Ljung-Box/autocorrelation diagnostics
- OLS regression tables with robust standard errors
- rolling AR(1) persistence
- decay/convergence table
- out-of-sample forecast comparison when implemented
