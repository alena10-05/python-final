"""
Synthetic multi-country COVID time series (Alen Barnes reference).
Maps to app schema: cumulative confirmed/deaths/recovered + daily new_* + active.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from data_manager import COLUMNS


def generate_synthetic_cases_dataframe() -> pd.DataFrame:
    """2020-01-22 → 2023-05-11, 10 countries — realistic waves (same logic as Barnes reference)."""
    np.random.seed(42)
    start = datetime(2020, 1, 22)
    end = datetime(2023, 5, 11)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    countries = {
        "USA": {"pop": 331_000_000, "peak": 900_000, "waves": [180, 360, 540, 720]},
        "India": {"pop": 1_380_000_000, "peak": 400_000, "waves": [300, 480, 600, 750]},
        "Brazil": {"pop": 215_000_000, "peak": 300_000, "waves": [240, 420, 580, 780]},
        "UK": {"pop": 67_000_000, "peak": 150_000, "waves": [200, 380, 560, 700]},
        "Germany": {"pop": 83_000_000, "peak": 120_000, "waves": [210, 390, 570, 710]},
        "France": {"pop": 67_000_000, "peak": 110_000, "waves": [190, 370, 550, 690]},
        "Italy": {"pop": 60_000_000, "peak": 100_000, "waves": [170, 350, 530, 670]},
        "Spain": {"pop": 47_000_000, "peak": 90_000, "waves": [175, 355, 535, 675]},
        "Canada": {"pop": 38_000_000, "peak": 50_000, "waves": [215, 395, 575, 715]},
        "Japan": {"pop": 125_000_000, "peak": 80_000, "waves": [230, 410, 590, 730]},
    }

    rows: list[dict] = []
    for country, cfg in countries.items():
        daily_cases = np.zeros(n)
        for wave_day in cfg["waves"]:
            sigma = 60 + np.random.randint(-10, 20)
            peak = cfg["peak"] * (0.6 + np.random.random() * 0.8)
            x = np.arange(n)
            wave = peak * np.exp(-0.5 * ((x - wave_day) / sigma) ** 2)
            daily_cases += wave
        noise = np.random.lognormal(0, 0.3, n)
        daily_cases = np.maximum(0, daily_cases * noise).astype(int)
        cum_confirmed = np.cumsum(daily_cases)

        cfr = 0.012 + np.random.random() * 0.008
        daily_deaths = (daily_cases * cfr * (0.8 + 0.4 * np.random.random(n))).astype(int)
        cum_deaths = np.cumsum(daily_deaths)

        recovery_lag = 14
        rr = 0.92 + np.random.random() * 0.05
        daily_recov = np.zeros(n, dtype=int)
        daily_recov[recovery_lag:] = (daily_cases[:-recovery_lag] * rr).astype(int)
        cum_recovered = np.cumsum(daily_recov)

        active = np.maximum(0, cum_confirmed - cum_deaths - cum_recovered)

        for i, d in enumerate(dates):
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "date": d.strftime("%Y-%m-%d"),
                    "country": country,
                    "region": "National",
                    "confirmed_cases": int(cum_confirmed[i]),
                    "new_cases": int(daily_cases[i]),
                    "deaths": int(cum_deaths[i]),
                    "new_deaths": int(daily_deaths[i]),
                    "recovered": int(cum_recovered[i]),
                    "active_cases": int(active[i]),
                    "tests_conducted": int(daily_cases[i] * 12 + np.random.randint(0, 5000)),
                    "hospitalized": int(daily_cases[i] * 0.05),
                    "critical": int(daily_cases[i] * 0.01),
                    "population": cfg["pop"],
                }
            )

    return pd.DataFrame(rows)[COLUMNS]
