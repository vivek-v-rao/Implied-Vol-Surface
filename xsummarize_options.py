#!/usr/bin/env python3
"""
Summarize option chain data from a CSV file.
"""

from __future__ import annotations

import sys

import pandas as pd

from option_summary import summarize_option_chain


def main(argv: list[str]) -> None:
    """CLI entrypoint."""
    if len(argv) < 2:
        print("usage: python xsummarize_options.py options.csv")
        sys.exit(1)
    infile = argv[1]
    df = pd.read_csv(infile)
    try:
        summary = summarize_option_chain(df, None)
    except ValueError as exc:
        print(f"error: {exc}")
        sys.exit(1)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
