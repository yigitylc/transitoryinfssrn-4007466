"""Full Streamlit app smoke coverage via ``streamlit.testing.v1.AppTest``.

Marked ``slow`` and excluded from the default ``pytest`` / ``pytest -q`` run
(see ``pyproject.toml``'s ``addopts``): a full-app render currently costs on
the order of a minute per sidebar combination (pre-existing, tracked
separately as a performance item; not addressed here), so this file is meant
to be run explicitly and deliberately rather than on every routine check:

    pytest -m slow tests/test_app_smoke.py

Post-H2 audit finding this file fixes: an earlier ad hoc AppTest sweep never
actually changed the sidebar widgets between iterations, so it silently
re-rendered one default combination 15 times and never exercised the other
14 -- including the exact combination that raises (see
``test_trader_report.py``'s ``test_full_sample_baseline_discloses_when_missing_cpi_zeroes_historical_evidence``
for the underlying report.py fix). ``test_sidebar_sweep_exercises_all_fifteen_combinations``
below drives the real sidebar radio/selectbox widgets through every baseline
x sample-mode combination and proves, via each widget's own post-run value,
that the change actually took effect.

Data is a monkeypatched, deterministic macro/market fixture (no network, no
dependence on a local FRED cache) so the sweep is reproducible in any
environment. See docs/02_DATA_CONTRACT.md for why a permanent 2025-10 CPI
gap belongs in the fixture: it is what makes full_sample's whole-column
baseline lineage mark the entire historical population ineligible for
``live_dashboard`` and ``max_history``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation import data as data_mod
from transitory_inflation import market_data as market_data_mod
from transitory_inflation.config import SAMPLE_MODES, resolve_sample_mode
from transitory_inflation.data import MacroDataLoadResult, build_base_frame
from transitory_inflation.features import BASELINE_META
from transitory_inflation.market_data import (
    MarketDataLoadResult,
    available_market_variables,
    build_market_close_frame,
    build_market_frame,
    latest_valid_dates_by_variable,
)

APP_PATH = "app/streamlit_app.py"

# 1981-01-31 gives every named sample mode its required 12-month pre-sample
# warmup ahead of the shared 1982-01-01 anchor; 2026-06-30 reaches past the
# permanent 2025-10 CPI gap so live_dashboard/max_history actually exercise
# it. Both bounds are load-bearing -- see docs/02_DATA_CONTRACT.md.
_DATES = pd.date_range("1981-01-31", "2026-06-30", freq="ME")
_GAP = pd.Timestamp("2025-10-31") == _DATES

_EXPECTED_SAMPLE_MODES = frozenset(SAMPLE_MODES)
_EXPECTED_BASELINES = frozenset(BASELINE_META)
_EXPECTED_COMBINATIONS = frozenset(
    (sample_mode, baseline)
    for sample_mode in _EXPECTED_SAMPLE_MODES
    for baseline in _EXPECTED_BASELINES
)
# full_sample's baseline value depends on every loaded row, so the permanent
# 2025-10 gap marks the whole historical population ineligible only where
# that gap is actually in-sample. paper_replication ends 2021-07, before the
# gap, so it is unaffected.
_KNOWN_ZERO_ELIGIBILITY_COMBINATIONS = frozenset(
    {("live_dashboard", "full_sample"), ("max_history", "full_sample")}
)


def _build_synthetic_warmup() -> pd.DataFrame:
    periods = len(_DATES)
    headline = 100.0 * (1.002 ** np.arange(periods, dtype=float))
    core = 95.0 * (1.0018 ** np.arange(periods, dtype=float))
    headline[_GAP] = np.nan
    core[_GAP] = np.nan
    return build_base_frame(
        pd.DataFrame({"date": _DATES, "CPIAUCSL": headline, "CPILFESL": core, "TB3MS": 3.0}),
        imputation_policy="observed_only",
    )


def _build_synthetic_market_raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    periods = len(_DATES)
    return pd.DataFrame(
        {
            "date": _DATES,
            "DGS2": 1.0 + 0.01 * np.arange(periods) + rng.normal(0, 0.05, periods),
            "DGS10": 2.0 + 0.008 * np.arange(periods) + rng.normal(0, 0.05, periods),
            "T5YIE": 2.0 + rng.normal(0, 0.1, periods),
            "T10YIE": 2.2 + rng.normal(0, 0.1, periods),
            "DFII5": 0.5 + rng.normal(0, 0.1, periods),
            "DFII10": 0.7 + rng.normal(0, 0.1, periods),
        }
    )


def _make_fake_macro_loader(warmup: pd.DataFrame):
    def _fake_macro_loader(
        mode: str, imputation_policy: str = "observed_only"
    ) -> MacroDataLoadResult:
        resolved = resolve_sample_mode(mode)
        start = pd.Timestamp(resolved.start_date) if resolved.start_date else warmup["date"].min()
        end = pd.Timestamp(resolved.end_date) if resolved.end_date else warmup["date"].max()
        sliced = warmup.loc[warmup["date"].between(start, end)].reset_index(drop=True)
        return MacroDataLoadResult(
            data=sliced,
            data_source_used="demo",
            live_fetch_status="ok (deterministic test fixture)",
            cache_file_used=None,
            api_key_configured=False,
            imputation_policy="observed_only",
            warmup_data=warmup,
        )

    return _fake_macro_loader


def _make_fake_market_loader(market_raw: pd.DataFrame):
    def _fake_market_loader(mode: str) -> MarketDataLoadResult:
        resolved = resolve_sample_mode(mode)
        closes = build_market_close_frame(
            market_raw, start_date=resolved.start_date, end_date=resolved.end_date
        )
        monthly = build_market_frame(
            market_raw, start_date=resolved.start_date, end_date=resolved.end_date
        )
        return MarketDataLoadResult(
            data=monthly,
            market_data_source_used="demo",
            market_live_fetch_status="ok (deterministic test fixture)",
            market_closes=closes,
            market_cache_file_used=None,
            available_market_variables=available_market_variables(closes),
            latest_valid_date_by_variable=latest_valid_dates_by_variable(closes),
            api_key_configured=False,
        )

    return _fake_market_loader


@pytest.fixture
def deterministic_app(monkeypatch: pytest.MonkeyPatch):
    """A running AppTest wired to deterministic, network-free data fixtures.

    Patches the two loader functions the app calls directly (rather than the
    lower-level FRED fetch/cache machinery) so the sweep is independent of
    network access and of whatever, if anything, happens to be cached under
    ``data/raw/`` on the machine running the tests. The ``streamlit.testing``
    import and the synthetic-frame construction both happen here rather than
    at module scope, so collecting this file (which happens even when the
    ``slow``-marked test below is deselected) stays cheap.
    """

    from streamlit.testing.v1 import AppTest

    warmup = _build_synthetic_warmup()
    market_raw = _build_synthetic_market_raw()
    monkeypatch.setattr(
        data_mod, "load_macro_data_for_mode_with_status", _make_fake_macro_loader(warmup)
    )
    monkeypatch.setattr(
        market_data_mod,
        "load_market_data_for_mode_with_status",
        _make_fake_market_loader(market_raw),
    )

    def _blocked_network_call(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError("network access is blocked in tests")

    monkeypatch.setattr("requests.get", _blocked_network_call)
    monkeypatch.setattr("requests.Session.request", _blocked_network_call)

    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.slow
def test_sidebar_sweep_exercises_all_fifteen_combinations(deterministic_app) -> None:
    """Drive every baseline x sample-mode combination through the real sidebar.

    Each iteration asserts the widgets' own post-run values equal the
    requested combination -- proof the sweep changed them, not just that the
    loop iterated -- and that the resulting set of exercised combinations is
    the full 5x3 Cartesian product. The two combinations where full_sample's
    baseline spans the permanent 2025-10 CPI gap must render the report's
    graceful disclosure warning with zero exceptions; every other
    full_sample combination must not show that warning.
    """

    at = deterministic_app
    assert len(at.sidebar.radio) == 1, "expected exactly one sidebar sample-mode radio"
    assert len(at.sidebar.selectbox) == 1, "expected exactly one sidebar baseline selectbox"

    exercised: set[tuple[str, str]] = set()

    for sample_mode in sorted(_EXPECTED_SAMPLE_MODES):
        for baseline in sorted(_EXPECTED_BASELINES):
            # Re-fetched fresh each iteration: at.run() rebuilds the element
            # tree, so a widget proxy captured before an earlier run() is
            # stale and must not be reused across reruns.
            at.sidebar.radio[0].set_value(sample_mode)
            at.sidebar.selectbox[0].set_value(baseline)
            at.run()

            # The widgets' own post-run state is the proof the sweep actually
            # moved them -- not merely that set_value()/run() were called.
            assert at.sidebar.radio[0].value == sample_mode
            assert at.sidebar.selectbox[0].value == baseline
            assert not at.exception, (
                f"{sample_mode} x {baseline} raised: {[e.value for e in at.exception]}"
            )
            exercised.add((sample_mode, baseline))

            warning_text = "\n".join(w.value for w in at.warning)
            gap_disclosed = "no observed-only eligible rows" in warning_text
            if (sample_mode, baseline) in _KNOWN_ZERO_ELIGIBILITY_COMBINATIONS:
                assert gap_disclosed, (
                    f"{sample_mode} x {baseline}: expected the full_sample zero-eligible-"
                    "historical-evidence disclosure warning; found none. Report tab "
                    "routing must disclose this case, not silently start passing."
                )
            elif baseline == "full_sample":
                assert not gap_disclosed, (
                    f"{sample_mode} x {baseline}: unexpectedly disclosed a zero-eligible-"
                    "evidence warning outside the known permanent-CPI-gap combinations."
                )

    assert exercised == _EXPECTED_COMBINATIONS
    assert len(exercised) == 15
