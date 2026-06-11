from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCES_DIR = PROJECT_ROOT / "references"


@dataclass(frozen=True)
class SampleMode:
    """Named date-range policy for loading macro data.

    ``start_date`` and ``end_date`` are inclusive bounds. ``None`` means
    unbounded on that side, i.e. use whatever FRED has available.
    """

    name: str
    start_date: str | None
    end_date: str | None
    purpose: str
    description: str


SAMPLE_MODES: dict[str, SampleMode] = {
    "paper_replication": SampleMode(
        name="paper_replication",
        start_date="1982-01-01",
        end_date="2021-07-31",
        purpose="Reproduce the paper only.",
        description=(
            "Fixed 1982-01 to 2021-07 sample for methodology replication. "
            "Ex-post view; not a live signal."
        ),
    ),
    "live_dashboard": SampleMode(
        name="live_dashboard",
        start_date="1982-01-01",
        end_date=None,
        purpose="Current macro signal using latest available FRED data.",
        description=(
            "1982-01 through the latest available FRED observation. "
            "Default mode for current macro interpretation."
        ),
    ),
    "max_history": SampleMode(
        name="max_history",
        start_date=None,
        end_date=None,
        purpose="Robustness test using earliest available FRED history.",
        description=(
            "Earliest available FRED history through the latest observation. "
            "A longer sample shifts percentiles and regime statistics, so this "
            "is a robustness view, not necessarily the default trading signal."
        ),
    ),
}

DEFAULT_SAMPLE_MODE = "live_dashboard"


def resolve_sample_mode(mode: SampleMode | str) -> SampleMode:
    """Return the SampleMode for a name, validating against the registry."""

    if isinstance(mode, SampleMode):
        return mode
    if mode not in SAMPLE_MODES:
        raise ValueError(f"Unknown sample mode: {mode!r}. Expected one of {sorted(SAMPLE_MODES)}")
    return SAMPLE_MODES[mode]


@dataclass(frozen=True)
class SeriesConfig:
    """FRED series choices used by the starter project."""

    cpi: str = "CPIAUCSL"  # CPI for All Urban Consumers: All Items in U.S. City Average
    tbill_3m: str = "TB3MS"  # 3-Month Treasury Bill Secondary Market Rate, monthly, since 1934
    core_cpi: str = "CPILFESL"  # Core CPI level
    pce: str = "PCEPI"  # PCE price index
    core_pce: str = "PCEPILFE"  # Core PCE price index
    breakeven_10y: str = "T10YIE"  # 10-Year Breakeven Inflation Rate, daily
    real_yield_10y: str = "DFII10"  # 10-Year TIPS real yield, daily
    nominal_yield_10y: str = "DGS10"  # 10-Year Treasury constant maturity, daily
    fed_funds: str = "FEDFUNDS"  # Effective fed funds rate, monthly


@dataclass(frozen=True)
class ResearchConfig:
    """Default research configuration."""

    sample_mode: str = DEFAULT_SAMPLE_MODE
    baseline_method: str = "rolling_36_shifted"
    tinf_windows: tuple[int, ...] = (4, 8, 12)
    rolling_rho_windows: tuple[int, ...] = (24, 30, 36, 48, 60)
    convergence_threshold: float = 0.95
    fred: SeriesConfig = field(default_factory=SeriesConfig)
