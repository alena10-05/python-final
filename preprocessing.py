"""Pandas-based preprocessing for COVID-19 tabular data (cleaning, typing, deduplication)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_manager import COLUMNS, NUMERIC_COLS, STRING_COLS

# Optional aliases when ingesting public CSVs (e.g. OWID-style exports)
COLUMN_ALIASES: dict[str, str] = {
    "location": "country",
    "iso_code": "region",
    "total_cases": "confirmed_cases",
    "new_cases_smoothed": "new_cases",
    "total_deaths": "deaths",
    "new_deaths_smoothed": "new_deaths",
    "total_recovered": "recovered",
}


@dataclass
class PreprocessOptions:
    strip_strings: bool = True
    coerce_numeric: bool = True
    parse_dates: bool = True
    drop_duplicates: bool = True
    duplicate_keys: tuple[str, ...] = ("date", "country", "region")
    fill_numeric_na: bool = True


@dataclass
class PreprocessResult:
    dataframe: pd.DataFrame
    messages: list[str] = field(default_factory=list)


def rename_known_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Map common alternate column names to the app schema."""
    lower = {c.lower(): c for c in df.columns}
    renames: dict[str, str] = {}
    for alias, target in COLUMN_ALIASES.items():
        if alias in df.columns and target not in df.columns:
            renames[alias] = target
        elif alias in lower and target not in df.columns:
            renames[lower.get(alias, alias)] = target
    if renames:
        df = df.rename(columns=renames)
    return df


def preprocess_covid_dataframe(df: pd.DataFrame, options: PreprocessOptions | None = None) -> PreprocessResult:
    """
    Clean and standardize a COVID case dataframe for storage and analysis.
    Uses Pandas for manipulation; NumPy where aggregate/array ops apply downstream.
    """
    options = options or PreprocessOptions()
    messages: list[str] = []
    out = df.copy()
    n0 = len(out)

    out = rename_known_aliases(out)

    for c in STRING_COLS:
        if c not in out.columns:
            out[c] = ""
    for c in NUMERIC_COLS:
        if c not in out.columns:
            out[c] = 0

    if options.strip_strings:
        for c in STRING_COLS:
            out[c] = out[c].astype(str).str.strip()
        messages.append("Stripped whitespace from text fields.")

    if options.parse_dates and "date" in out.columns:
        parsed = pd.to_datetime(out["date"], errors="coerce")
        bad = parsed.isna() & out["date"].astype(str).str.strip().ne("")
        if bad.any():
            messages.append(f"Dropped {int(bad.sum())} row(s) with unparseable dates.")
        out["date"] = parsed.dt.strftime("%Y-%m-%d")
        out = out.dropna(subset=["date"])

    if options.coerce_numeric:
        for c in NUMERIC_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        if options.fill_numeric_na:
            for c in NUMERIC_COLS:
                out[c] = out[c].fillna(0).astype(np.float64)
                out[c] = np.floor(out[c].to_numpy()).astype(np.int64)
        messages.append("Coerced numeric columns to integers (NaN → 0 when filling).")
    else:
        for c in NUMERIC_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.int64)

    if options.drop_duplicates and options.duplicate_keys:
        keys = [k for k in options.duplicate_keys if k in out.columns]
        if keys:
            before = len(out)
            out = out.drop_duplicates(subset=keys, keep="last")
            dropped = before - len(out)
            if dropped:
                messages.append(f"Removed {dropped} duplicate row(s) on {keys}.")

    # Ensure required string fields for DB
    out = out.replace({np.nan: None})
    for c in ("country", "region"):
        out[c] = out[c].fillna("").astype(str)

    messages.append(f"Rows: {n0} → {len(out)}.")

    for c in COLUMNS:
        if c == "id":
            continue
        if c not in out.columns:
            out[c] = 0 if c in NUMERIC_COLS else ""
    if "id" not in out.columns:
        out["id"] = ""
    return PreprocessResult(dataframe=out[COLUMNS], messages=messages)
