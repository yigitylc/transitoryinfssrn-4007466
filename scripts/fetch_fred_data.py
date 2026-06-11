from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from transitory_inflation.config import SAMPLE_MODES
from transitory_inflation.data import load_macro_data_for_mode, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the base FRED macro dataset for a named sample mode."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(SAMPLE_MODES),
        default="max_history",
        help=(
            "Sample mode date range to fetch. Default is max_history because the "
            "raw cache should be the superset any other mode can be sliced from."
        ),
    )
    args = parser.parse_args()

    df = load_macro_data_for_mode(args.mode)
    out = save_dataset(df, PROJECT_ROOT / "data" / "raw" / f"fred_base_macro_{args.mode}.csv")
    span = f"{df['date'].min().date()} -> {df['date'].max().date()}"
    print(f"Saved {len(df):,} rows ({args.mode}, {span}) to {out}")


if __name__ == "__main__":
    main()
