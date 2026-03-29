"""Matplotlib figures (Barnes-style) for embedding in Tkinter — not Agg file export."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from analytics_core import compute_barnes_statistics, preprocess_for_analytics
from theme import PALETTE


def _style_axes(ax) -> None:
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["muted"])
    for s in ax.spines.values():
        s.set_color(PALETTE["border"])
    ax.grid(True, alpha=0.25, color=PALETTE["border"])


def figure_global_trends(df: pd.DataFrame) -> Figure:
    d = preprocess_for_analytics(df)
    world = d.groupby("date")[["confirmed", "deaths_cum", "recovered_cum", "active"]].sum().reset_index()

    fig = Figure(figsize=(11, 7), facecolor=PALETTE["bg"])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0.38, left=0.08, right=0.96, top=0.94, bottom=0.08)

    ax1.fill_between(world["date"], world["confirmed"] / 1e6, alpha=0.15, color=PALETTE["accent1"])
    ax1.plot(world["date"], world["confirmed"] / 1e6, color=PALETTE["accent1"], lw=2, label="Confirmed")
    ax1.fill_between(world["date"], world["deaths_cum"] / 1e6, alpha=0.25, color=PALETTE["accent2"])
    ax1.plot(world["date"], world["deaths_cum"] / 1e6, color=PALETTE["accent2"], lw=2, label="Deaths")
    ax1.fill_between(world["date"], world["recovered_cum"] / 1e6, alpha=0.15, color=PALETTE["accent3"])
    ax1.plot(world["date"], world["recovered_cum"] / 1e6, color=PALETTE["accent3"], lw=2, label="Recovered")
    ax1.set_title("Global cumulative trends (millions)", color=PALETTE["text"], fontsize=12)
    ax1.set_ylabel("Millions", color=PALETTE["muted"], fontsize=9)
    ax1.legend(
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8,
    )
    _style_axes(ax1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    wd = d.groupby("date")[["daily_cases", "daily_deaths"]].sum().reset_index()
    wd["roll7_cases"] = wd["daily_cases"].rolling(7).mean()
    wd["roll7_deaths"] = wd["daily_deaths"].rolling(7).mean()

    ax2.bar(wd["date"], wd["daily_cases"] / 1e3, color=PALETTE["accent1"], alpha=0.35, width=1.0, label="Daily cases")
    ax2.plot(wd["date"], wd["roll7_cases"] / 1e3, color=PALETTE["accent1"], lw=2, label="7-day avg (cases)")
    ax2_r = ax2.twinx()
    ax2_r.plot(wd["date"], wd["roll7_deaths"] / 1e3, color=PALETTE["accent2"], lw=1.5, linestyle="--", label="7-day avg (deaths)")
    ax2_r.set_ylabel("Deaths (K)", color=PALETTE["muted"], fontsize=9)
    ax2_r.tick_params(colors=PALETTE["muted"])
    ax2.set_title("Daily incidence & 7-day rolling averages (thousands)", color=PALETTE["text"], fontsize=11)
    ax2.set_ylabel("Cases (K)", color=PALETTE["muted"], fontsize=9)
    ax2.legend(loc="upper left", facecolor=PALETTE["surface"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"], fontsize=7)
    _style_axes(ax2)
    _style_axes(ax2_r)
    ax2.spines["top"].set_visible(False)

    return fig


def figure_country_bars(stats: dict) -> Figure:
    cs = stats["countries"]
    countries = list(cs.keys())
    confirmed = [cs[c]["confirmed"] / 1e6 for c in countries]
    deaths = [cs[c]["deaths"] / 1e6 for c in countries]
    cfr = [cs[c]["cfr"] for c in countries]

    idx = np.argsort(confirmed)[::-1]
    countries = [countries[i] for i in idx]
    confirmed = [confirmed[i] for i in idx]
    deaths = [deaths[i] for i in idx]
    cfr = [cfr[i] for i in idx]

    fig = Figure(figsize=(12, 5.5), facecolor=PALETTE["bg"])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97, top=0.9, bottom=0.2)

    x = np.arange(len(countries))
    w = 0.38
    ax1.bar(x - w / 2, confirmed, w, color=PALETTE["accent1"], alpha=0.88, label="Confirmed")
    ax1.bar(x + w / 2, deaths, w, color=PALETTE["accent2"], alpha=0.88, label="Deaths")
    ax1.set_xticks(x)
    ax1.set_xticklabels(countries, rotation=35, ha="right", fontsize=8, color=PALETTE["muted"])
    ax1.set_ylabel("Millions", color=PALETTE["muted"])
    ax1.set_title("Total confirmed vs deaths", color=PALETTE["text"], fontsize=11)
    ax1.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"], fontsize=8)
    _style_axes(ax1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    colors = [PALETTE["accent2"] if c > np.mean(cfr) else PALETTE["accent3"] for c in cfr]
    ax2.barh(countries, cfr, color=colors, alpha=0.88)
    ax2.axvline(np.mean(cfr), color=PALETTE["accent4"], linestyle="--", lw=1.2, label=f"Mean CFR {np.mean(cfr):.2f}%")
    ax2.set_xlabel("Case fatality rate (%)", color=PALETTE["muted"])
    ax2.set_title("CFR by country", color=PALETTE["text"], fontsize=11)
    ax2.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"], fontsize=8)
    _style_axes(ax2)
    ax2.spines["top"].set_visible(False)

    return fig


def figure_wave_panel(df: pd.DataFrame) -> Figure:
    d = preprocess_for_analytics(df)
    top3 = ["USA", "India", "Brazil"]
    fig = Figure(figsize=(11, 8), facecolor=PALETTE["bg"])
    ax_a = fig.add_subplot(311)
    ax_b = fig.add_subplot(312, sharex=ax_a)
    ax_c = fig.add_subplot(313, sharex=ax_a)
    axes = (ax_a, ax_b, ax_c)
    fig.subplots_adjust(hspace=0.22, left=0.1, right=0.97, top=0.94, bottom=0.08)
    colors_list = [PALETTE["accent1"], PALETTE["accent4"], PALETTE["accent3"]]

    for ax, country, col in zip(axes, top3, colors_list):
        c_df = d[d["country"] == country].sort_values("date")
        y = c_df["roll7_cases"].fillna(0) / 1e3
        ax.fill_between(c_df["date"], y, alpha=0.28, color=col)
        ax.plot(c_df["date"], y, color=col, lw=2)
        ax.set_ylabel(f"{country}\n(K)", color=PALETTE["muted"], fontsize=9)
        _style_axes(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        peak = c_df["roll7_cases"].max()
        if pd.notna(peak) and peak > 0:
            peak_date = c_df.loc[c_df["roll7_cases"].idxmax(), "date"]
            ax.annotate(
                f"Peak {peak/1e3:.0f}K",
                xy=(peak_date, peak / 1e3),
                xytext=(25, 5),
                textcoords="offset points",
                color=col,
                fontsize=8,
            )

    axes[0].set_title("Wave analysis — 7-day rolling average (top 3)", color=PALETTE["text"], fontsize=12)
    return fig


def figure_heatmap_monthly(df: pd.DataFrame) -> Figure:
    d = preprocess_for_analytics(df)
    d = d.copy()
    d["month"] = d["date"].dt.to_period("M")
    monthly = d.groupby(["country", "month"])["daily_cases"].sum().reset_index()
    pivot = monthly.pivot(index="country", columns="month", values="daily_cases").fillna(0) / 1000.0
    cols = pivot.columns[::3]
    if len(cols) == 0:
        cols = pivot.columns
    pivot = pivot[cols]

    fig = Figure(figsize=(12, 5), facecolor=PALETTE["bg"])
    ax = fig.add_subplot(111)
    cmap = LinearSegmentedColormap.from_list("covid", ["#0D1117", "#1a3a5c", PALETTE["accent1"], PALETTE["accent2"], "#ff4d4d"])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color=PALETTE["muted"])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right", fontsize=7, color=PALETTE["muted"])
    ax.set_title("Monthly new cases heatmap (thousands, every 3rd month)", color=PALETTE["text"], fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Cases (K)")
    ax.set_facecolor(PALETTE["surface"])
    return fig


def figure_scatter_burden(df: pd.DataFrame, stats: dict) -> Figure:
    cs = stats["countries"]
    countries = list(cs.keys())
    x = np.array([cs[c]["cases_per_1m"] for c in countries])
    y = np.array([cs[c]["deaths_per_1m"] for c in countries])
    cfr = np.array([cs[c]["cfr"] for c in countries])
    sizes = np.array([cs[c]["confirmed"] for c in countries]) / 1e5

    fig = Figure(figsize=(8, 6), facecolor=PALETTE["bg"])
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, s=np.clip(sizes, 20, 800), c=cfr, cmap="RdYlGn_r", alpha=0.88, edgecolors=PALETTE["border"], linewidths=1.0)
    for i, country in enumerate(countries):
        ax.annotate(country, (x[i], y[i]), textcoords="offset points", xytext=(5, 4), fontsize=8, color=PALETTE["text"])
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 80)
        ax.plot(xl, m * xl + b, "--", color=PALETTE["accent4"], lw=1.5)
    ax.set_xlabel("Cases per million", color=PALETTE["muted"])
    ax.set_ylabel("Deaths per million", color=PALETTE["muted"])
    ax.set_title("Population-normalised burden (bubble size ∝ confirmed)", color=PALETTE["text"], fontsize=11)
    fig.colorbar(sc, ax=ax, label="CFR %")
    _style_axes(ax)
    ax.set_facecolor(PALETTE["surface"])
    return fig


def embed_figure(parent, fig: Figure) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.configure(bg=PALETTE["bg"])
    return canvas
