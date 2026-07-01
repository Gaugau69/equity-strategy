"""
main.py
-------
Entry point wired to the strategy CLI.

Usage
-----
    python main.py                        # simulated data (default)
    python main.py --csv data/prices.csv  # real CSV data
    python main.py --tc 5 --top_n 50      # custom params
    python main.py --help                 # all options

Delegates to scripts/run_strategy.py which contains the full pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.run_strategy import main  # noqa: E402

if __name__ == "__main__":
    main()
