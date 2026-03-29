"""Export a static HTML dashboard (Barnes-style) with KPI tables + embedded chart PNGs."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from analytics_charts import (
    figure_country_bars,
    figure_global_trends,
    figure_heatmap_monthly,
    figure_scatter_burden,
    figure_wave_panel,
)
from analytics_core import compute_barnes_statistics, fmt_compact
from theme import PALETTE


def export_dashboard_html(path: str | Path, dm) -> None:
    """Write HTML report: global KPIs, country table, SQL-style summary, embedded matplotlib figures."""
    from data_manager import DataManager

    assert isinstance(dm, DataManager)
    df = dm.dataframe()
    path = Path(path)
    stats = compute_barnes_statistics(df) if len(df) else {"global": {}, "countries": {}}
    g = stats.get("global", {})

    figures: list[tuple[str, object]] = []
    if len(df):
        figures.append(("Global trends", figure_global_trends(df)))
        figures.append(("Country comparison", figure_country_bars(stats)))
        figures.append(("Wave analysis", figure_wave_panel(df)))
        figures.append(("Monthly heatmap", figure_heatmap_monthly(df)))
        figures.append(("Burden scatter", figure_scatter_burden(df, stats)))

    img_tags = []
    import matplotlib.pyplot as plt

    for title, fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=PALETTE["bg"], edgecolor="none")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        img_tags.append(f'<h3 style="color:{PALETTE["accent1"]};font-family:Consolas,monospace">{title}</h3>')
        img_tags.append(f'<p><img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid {PALETTE["border"]};border-radius:8px"/></p>')

    rows = ""
    for c, d in sorted(stats.get("countries", {}).items(), key=lambda x: x[1]["confirmed"], reverse=True):
        rows += f"""<tr>
<td>{c}</td><td>{fmt_compact(d['confirmed'])}</td><td>{fmt_compact(d['deaths'])}</td>
<td>{fmt_compact(d['recovered'])}</td><td>{fmt_compact(d['active'])}</td>
<td>{d['cfr']}%</td><td>{d['recovery_rate']}%</td><td>{d['peak_daily']:,}</td></tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/><title>COVID-19 Dashboard Export</title>
<style>
body{{background:{PALETTE['bg']};color:{PALETTE['text']};font-family:Segoe UI,sans-serif;padding:32px;max-width:1200px;margin:0 auto;}}
h1{{font-size:28px;background:linear-gradient(90deg,{PALETTE['text']},{PALETTE['accent1']});-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px;}}
th,td{{border:1px solid {PALETTE['border']};padding:8px;text-align:right;}}
th{{background:{PALETTE['surface']};color:{PALETTE['muted']};text-align:left;}}
td:first-child{{text-align:left;font-weight:600;}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:24px 0;}}
.kpi div{{background:{PALETTE['surface']};border:1px solid {PALETTE['border']};border-radius:8px;padding:16px;}}
.kpi small{{color:{PALETTE['muted']};text-transform:uppercase;font-size:10px;}}
.kpi strong{{font-size:22px;display:block;margin-top:8px;}}
</style></head><body>
<h1>COVID-19 Data Analysis Dashboard</h1>
<p style="color:{PALETTE['muted']}">Exported report · Healthcare informatics · NumPy · Pandas · Matplotlib · SQLite</p>
<div class="kpi">
<div><small>Total confirmed (latest)</small><strong>{fmt_compact(g.get('total_confirmed',0))}</strong></div>
<div><small>Total deaths</small><strong>{fmt_compact(g.get('total_deaths',0))}</strong></div>
<div><small>Global CFR</small><strong>{g.get('global_cfr',0)}%</strong></div>
<div><small>Recovered</small><strong>{fmt_compact(g.get('total_recovered',0))}</strong></div>
<div><small>Recovery rate</small><strong>{g.get('global_recovery',0)}%</strong></div>
<div><small>Active (latest)</small><strong>{fmt_compact(g.get('total_active',0))}</strong></div>
<div><small>Countries</small><strong>{g.get('countries_tracked',0)}</strong></div>
<div><small>Date range</small><strong style="font-size:12px">{g.get('date_range','—')}</strong></div>
</div>
<h2 style="color:{PALETTE['accent1']};font-size:14px;font-family:Consolas,monospace">Country summary</h2>
<table><thead><tr>
<th>Country</th><th>Confirmed</th><th>Deaths</th><th>Recovered</th><th>Active</th><th>CFR</th><th>Recovery %</th><th>Peak daily</th>
</tr></thead><tbody>{rows or '<tr><td colspan="8">No data</td></tr>'}</tbody></table>
<h2 style="color:{PALETTE['accent1']};margin-top:32px;font-size:14px;font-family:Consolas,monospace">Visualisations</h2>
{chr(10).join(img_tags)}
</body></html>"""
    path.write_text(html, encoding="utf-8")
