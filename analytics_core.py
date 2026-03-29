"""Preprocessing + global / per-country statistics (Barnes-style metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess_for_analytics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["country", "date"])
    d["daily_cases"] = d["new_cases"].astype(float)
    d["daily_deaths"] = d["new_deaths"].astype(float)
    d["confirmed"] = d["confirmed_cases"].astype(float)
    d["deaths_cum"] = d["deaths"].astype(float)
    d["recovered_cum"] = d["recovered"].astype(float)
    d["active"] = d["active_cases"].astype(float)
    d["cfr"] = np.where(d["confirmed"] > 0, d["deaths_cum"] / d["confirmed"] * 100.0, 0.0)
    d["recovery_rate"] = np.where(d["confirmed"] > 0, d["recovered_cum"] / d["confirmed"] * 100.0, 0.0)
    d["cases_per_1m"] = d["confirmed"] / (d["population"].replace(0, np.nan) / 1_000_000.0)
    d["deaths_per_1m"] = d["deaths_cum"] / (d["population"].replace(0, np.nan) / 1_000_000.0)
    d["roll7_cases"] = d.groupby("country")["daily_cases"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    d["roll7_deaths"] = d.groupby("country")["daily_deaths"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    return d


def compute_barnes_statistics(df: pd.DataFrame) -> dict:
    """Latest snapshot per country + distributional stats on daily incidence."""
    d = preprocess_for_analytics(df)
    latest = d.sort_values("date").groupby("country").last().reset_index()

    stats: dict = {
        "global": {
            "total_confirmed": int(latest["confirmed"].sum()),
            "total_deaths": int(latest["deaths_cum"].sum()),
            "total_recovered": int(latest["recovered_cum"].sum()),
            "total_active": int(latest["active"].sum()),
            "global_cfr": round(float(latest["deaths_cum"].sum() / max(latest["confirmed"].sum(), 1)) * 100, 3),
            "global_recovery": round(float(latest["recovered_cum"].sum() / max(latest["confirmed"].sum(), 1)) * 100, 2),
            "countries_tracked": int(latest["country"].nunique()),
            "date_range": f"{d['date'].min().date()}  →  {d['date'].max().date()}",
        }
    }

    country_stats: dict = {}
    for _, row in latest.iterrows():
        c = row["country"]
        c_df = d[d["country"] == c]
        daily = c_df["daily_cases"].values
        country_stats[c] = {
            "confirmed": int(row["confirmed"]),
            "deaths": int(row["deaths_cum"]),
            "recovered": int(row["recovered_cum"]),
            "active": int(row["active"]),
            "cfr": round(float(row["cfr"]), 3),
            "recovery_rate": round(float(row["recovery_rate"]), 2),
            "cases_per_1m": round(float(row["cases_per_1m"]), 1),
            "deaths_per_1m": round(float(row["deaths_per_1m"]), 1),
            "peak_daily": int(np.max(daily)) if len(daily) else 0,
            "mean_daily": round(float(np.mean(daily)), 1),
            "std_daily": round(float(np.std(daily)), 1),
            "median_daily": round(float(np.median(daily)), 1),
        }

    stats["countries"] = country_stats
    return stats


def fmt_compact(n: float | int) -> str:
    n = float(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"
