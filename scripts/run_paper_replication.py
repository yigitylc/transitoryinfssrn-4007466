from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from transitory_inflation.data import load_macro_data_for_mode
from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.models import (
    correlation_matrix,
    decay_summaries_for_windows,
    run_paper_style_regressions,
    summary_stats,
)


def main() -> None:
    # Paper audit only: fixed 1982-01..2021-07 sample. The slice happens at load
    # time so the full_sample baseline mean cannot see post-2021 data.
    raw = load_macro_data_for_mode("paper_replication")
    df = add_transitory_inflation_features(raw, baseline_method="full_sample")

    print(
        f"PAPER REPLICATION SAMPLE: {raw['date'].min().date()} -> {raw['date'].max().date()}"
        f" ({len(raw)} months, baseline=full_sample, ex-post)"
    )

    cols = ["inflation_yoy", "tinf_4m", "tinf_8m", "tinf_12m", "tbill_3m"]

    print("\nSUMMARY STATS")
    print(summary_stats(df, cols).to_string(index=False))

    print("\nCORRELATION MATRIX")
    print(correlation_matrix(df, cols).round(3).to_string())

    print("\nREGRESSION TABLE")
    print(run_paper_style_regressions(df).round(4).to_string(index=False))

    print("\nDECAY SUMMARY")
    _, decay = decay_summaries_for_windows(df, windows=(24, 30), value_col="tinf_4m")
    print(decay.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
