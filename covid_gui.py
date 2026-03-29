"""
COVID-19 Data Analysis System — Tkinter GUI (Barnes-style dark dashboard + Matplotlib).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Literal

import numpy as np
import pandas as pd

from analytics_charts import (
    embed_figure,
    figure_country_bars,
    figure_global_trends,
    figure_heatmap_monthly,
    figure_scatter_burden,
    figure_wave_panel,
)
from analytics_core import compute_barnes_statistics, fmt_compact
from data_manager import DataManager, growth_rate
from preprocessing import PreprocessOptions
from theme import PALETTE


def _np_max(a):
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.max(arr))


def _np_min(a):
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.min(arr))


from data_manager import np_mean, np_median, np_std, np_sum


def fmt_num(n: float | int) -> str:
    return f"{int(n):,}" if float(n).is_integer() else f"{n:,.2f}"


MetricKind = Literal["confirmed_cases", "deaths", "recovered"]


class LayeredWindowMixin:
    """Top / middle / bottom frames (GitHub-dark, Barnes reference)."""

    def _build_layers(self, parent: tk.Widget, title: str, subtitle: str = "") -> tuple[tk.Frame, tk.Frame, tk.Frame]:
        top = tk.Frame(parent, bg=PALETTE["surface"], pady=10, padx=14, highlightthickness=1, highlightbackground=PALETTE["border"])
        top.pack(fill=tk.X, side=tk.TOP)
        tk.Label(top, text=title, font=("Segoe UI", 13, "bold"), fg=PALETTE["text"], bg=PALETTE["surface"]).pack(anchor=tk.W)
        if subtitle:
            tk.Label(top, text=subtitle, font=("Segoe UI", 9), fg=PALETTE["muted"], bg=PALETTE["surface"]).pack(anchor=tk.W)

        middle = tk.Frame(parent, bg=PALETTE["bg"])
        middle.pack(fill=tk.BOTH, expand=True)

        bottom = tk.Frame(parent, bg=PALETTE["surface"], pady=6, padx=10, highlightthickness=1, highlightbackground=PALETTE["border"])
        bottom.pack(fill=tk.X, side=tk.BOTTOM)
        return top, middle, bottom


class CovidMainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("COVID-19 Data Analysis Dashboard")
        self.geometry("1100x820")
        self.minsize(900, 680)
        self.configure(bg=PALETTE["bg"])

        self.dm = DataManager()
        self._windows: dict[str, tk.Toplevel | None] = {
            "entry": None,
            "viz": None,
            "stats": None,
            "search": None,
            "reports": None,
            "pipeline": None,
        }

        self._build_menubar()
        self._build_main_ui()
        self._refresh_dashboard()

    def _build_main_ui(self) -> None:
        hero = tk.Frame(self, bg=PALETTE["bg"], pady=20, padx=28)
        hero.pack(fill=tk.X)
        hero_inner = tk.Frame(hero, bg=PALETTE["bg"])
        hero_inner.pack(fill=tk.X)
        tk.Label(
            hero_inner,
            text="MICRO PROJECT 03  ·  CO3 CO4 CO5",
            font=("Consolas", 10),
            fg=PALETTE["accent1"],
            bg=PALETTE["bg"],
        ).pack(anchor=tk.W)
        tk.Label(
            hero_inner,
            text="COVID-19 Data Analysis Dashboard",
            font=("Segoe UI", 26, "bold"),
            fg=PALETTE["text"],
            bg=PALETTE["bg"],
        ).pack(anchor=tk.W, pady=(6, 4))
        tk.Label(
            hero_inner,
            text="NumPy · Pandas · SQLite · Matplotlib · Tkinter — healthcare informatics & data-driven analytics",
            font=("Segoe UI", 11),
            fg=PALETTE["muted"],
            bg=PALETTE["bg"],
        ).pack(anchor=tk.W)
        meta = tk.Frame(hero_inner, bg=PALETTE["bg"])
        meta.pack(anchor=tk.W, pady=(14, 0))
        self._meta_db = tk.Label(
            meta,
            text="",
            font=("Consolas", 9),
            fg=PALETTE["muted"],
            bg=PALETTE["surface"],
            padx=12,
            pady=6,
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        self._meta_db.pack(side=tk.LEFT, padx=(0, 10))

        mid = tk.Frame(self, bg=PALETTE["bg"], padx=24, pady=12)
        mid.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            mid,
            text="01 — GLOBAL KPI OVERVIEW (latest per country)",
            font=("Consolas", 9),
            fg=PALETTE["muted"],
            bg=PALETTE["bg"],
        ).pack(anchor=tk.W, pady=(0, 10))

        stats_fr = tk.Frame(mid, bg=PALETTE["bg"])
        stats_fr.pack(fill=tk.X, pady=(0, 14))
        self._stat_labels: dict[str, tk.Label] = {}
        names = [
            ("total_records", "DATA ROWS", PALETTE["accent1"]),
            ("total_cases", "TOTAL CONFIRMED", PALETTE["accent1"]),
            ("total_deaths", "TOTAL DEATHS", PALETTE["accent2"]),
            ("cfr_pct", "GLOBAL CFR", PALETTE["accent2"]),
            ("total_recovered", "RECOVERED", PALETTE["accent3"]),
            ("recovery_pct", "RECOVERY %", PALETTE["accent3"]),
            ("total_active", "ACTIVE CASES", PALETTE["accent4"]),
            ("countries", "COUNTRIES", PALETTE["purple"]),
        ]
        for i, (key, label, color) in enumerate(names):
            r, c = divmod(i, 4)
            f = tk.Frame(stats_fr, bg=PALETTE["surface"], highlightthickness=1, highlightbackground=PALETTE["border"])
            f.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            stats_fr.columnconfigure(c, weight=1)
            tk.Label(f, text=label, font=("Consolas", 8), fg=PALETTE["muted"], bg=PALETTE["surface"]).pack(anchor=tk.W, pady=(10, 4), padx=12)
            lb = tk.Label(f, text="—", font=("Segoe UI", 17, "bold"), fg=PALETTE["text"], bg=PALETTE["surface"])
            lb.pack(anchor=tk.W, padx=12, pady=(0, 10))
            tk.Frame(f, bg=color, height=3).pack(fill=tk.X, side=tk.BOTTOM)
            self._stat_labels[key] = lb

        nav = tk.LabelFrame(
            mid,
            text="Modules",
            font=("Consolas", 10, "bold"),
            bg=PALETTE["bg"],
            fg=PALETTE["text"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            labelanchor=tk.NW,
        )
        nav.pack(fill=tk.BOTH, expand=True)
        buttons = [
            ("Data Entry", "entry", PALETTE["accent1"], "CRUD case records"),
            ("Visualization", "viz", PALETTE["accent3"], "Matplotlib · global trends · waves · heatmap"),
            ("Statistics", "stats", PALETTE["purple"], "NumPy / Pandas · per-country blocks"),
            ("Search & Filter", "search", PALETTE["accent4"], "Query & narrow results"),
            ("Reports", "reports", PALETTE["accent2"], "Summary reports (SQLite JSON)"),
            ("Data pipeline & SQL", "pipeline", "#39C5CF", "Ingest · preprocess · SELECT"),
        ]
        for name, key, color, desc in buttons:
            row = tk.Frame(nav, bg=PALETTE["bg"])
            row.pack(fill=tk.X, padx=10, pady=6)
            b = tk.Button(
                row,
                text=name,
                font=("Segoe UI", 10, "bold"),
                fg=PALETTE["text"],
                bg=color,
                activebackground=color,
                relief=tk.FLAT,
                padx=14,
                pady=7,
                command=lambda k=key: self._open_secondary(k),
            )
            b.pack(side=tk.LEFT)
            tk.Label(row, text=desc, font=("Segoe UI", 9), fg=PALETTE["muted"], bg=PALETTE["bg"]).pack(side=tk.LEFT, padx=14)

        about = tk.Frame(mid, bg=PALETTE["surface"], highlightthickness=1, highlightbackground=PALETTE["border"], pady=10, padx=12)
        about.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            about,
            text=(
                "Load “full synthetic dataset” from the Data menu for Barnes-style multi-year time series (~12k rows). "
                "KPIs aggregate the latest cumulative values per country. Export HTML reproduces the reference dashboard layout."
            ),
            font=("Segoe UI", 9),
            fg=PALETTE["muted"],
            bg=PALETTE["surface"],
            wraplength=1000,
            justify=tk.LEFT,
        ).pack(fill=tk.X)

        self._bottom = tk.Frame(self, bg=PALETTE["surface"], pady=8, padx=14, highlightthickness=1, highlightbackground=PALETTE["border"])
        self._bottom.pack(fill=tk.X, side=tk.BOTTOM)
        self._status = tk.Label(
            self._bottom,
            text="Ready",
            font=("Consolas", 9),
            fg=PALETTE["muted"],
            bg=PALETTE["surface"],
            anchor=tk.W,
        )
        self._status.pack(fill=tk.X)

    def set_status(self, msg: str) -> None:
        self._status.config(text=msg)

    def _refresh_dashboard(self) -> None:
        s = self.dm.dashboard_stats()
        self._meta_db.config(text=f"SQLite  →  {self.dm.db_path}")
        disp = {
            "total_records": str(s["total_records"]),
            "total_cases": fmt_compact(s["total_cases"]),
            "total_deaths": fmt_compact(s["total_deaths"]),
            "cfr_pct": f"{s['cfr_pct']}%",
            "total_recovered": fmt_compact(s["total_recovered"]),
            "recovery_pct": f"{s['recovery_pct']}%",
            "total_active": fmt_compact(s["total_active"]),
            "countries": str(s["countries"]),
        }
        for k, v in disp.items():
            self._stat_labels[k].config(text=v)

    def _open_secondary(self, key: str) -> None:
        w = self._windows.get(key)
        if w is not None and w.winfo_exists():
            w.lift()
            w.focus_force()
            return
        ctor: dict[str, Callable[[], tk.Toplevel]] = {
            "entry": lambda: DataEntryWindow(self, self.dm, self._on_data_changed),
            "viz": lambda: VisualizationWindow(self, self.dm),
            "stats": lambda: StatisticsWindow(self, self.dm),
            "search": lambda: SearchFilterWindow(self, self.dm),
            "reports": lambda: ReportsWindow(self, self.dm),
            "pipeline": lambda: DatabasePipelineWindow(self, self.dm, self._on_data_changed),
        }
        win = ctor[key]()
        self._windows[key] = win

        def on_destroy(_: tk.Event | None = None) -> None:
            self._windows[key] = None

        win.bind("<Destroy>", on_destroy)

    def _on_data_changed(self) -> None:
        self._refresh_dashboard()
        self.set_status("Data updated.")

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self)
        data_menu = tk.Menu(menubar, tearoff=0)
        data_menu.add_command(label="Load full synthetic dataset (Barnes-style, ~12k rows)…", command=self._menu_seed_synthetic)
        data_menu.add_separator()
        data_menu.add_command(label="Import CSV (append to database)…", command=self._menu_import_csv)
        data_menu.add_command(label="Export all cases to CSV…", command=self._menu_export_csv)
        data_menu.add_separator()
        data_menu.add_command(label="Export HTML dashboard (like reference doc)…", command=self._menu_export_html)
        data_menu.add_separator()
        data_menu.add_command(label="Open data pipeline & SQL window…", command=lambda: self._open_secondary("pipeline"))
        menubar.add_cascade(label="Data", menu=data_menu)
        self.config(menu=menubar)

    def _menu_import_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            n, msgs = self.dm.ingest_csv(path, append=True)
            self._refresh_dashboard()
            self.set_status(f"Ingested {n} row(s) from CSV.")
            messagebox.showinfo("Import complete", f"Rows inserted: {n}\n\n" + "\n".join(msgs))
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def _menu_export_csv(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.dm.export_cases_csv(path)
            self.set_status(f"Exported cases to {path}")
            messagebox.showinfo("Export", "Cases exported successfully.")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _menu_seed_synthetic(self) -> None:
        if not messagebox.askyesno(
            "Synthetic dataset",
            "Replace ALL case rows with the full multi-country synthetic time series (2020–2023)?\n\n"
            "This matches the Alen Barnes reference (~12,000 rows). Reports table is kept.",
        ):
            return
        try:
            n = self.dm.seed_synthetic_full()
            self._refresh_dashboard()
            self.set_status(f"Loaded {n} synthetic rows.")
            messagebox.showinfo("Done", f"Database now contains {n:,} rows.\nOpen Visualization for Matplotlib charts.")
        except Exception as e:
            messagebox.showerror("Failed", str(e))

    def _menu_export_html(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML", "*.html")])
        if not path:
            return
        try:
            from html_export import export_dashboard_html

            export_dashboard_html(path, self.dm)
            self.set_status(f"HTML saved: {path}")
            messagebox.showinfo("Export", "Dashboard HTML written. Open it in a browser.")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))


class DataEntryWindow(tk.Toplevel, LayeredWindowMixin):
    def __init__(self, master: tk.Tk, dm: DataManager, on_change: Callable[[], None]) -> None:
        super().__init__(master)
        self.title("Data Entry — COVID-19")
        self.geometry("900x640")
        self.dm = dm
        self._on_change = on_change
        self._editing_id: str | None = None

        _, middle, bottom = self._build_layers(self, "Data Entry", "Add, edit, or delete case records")

        self._status = tk.Label(bottom, text="", font=("Segoe UI", 9), fg="#334155", bg="#e2e8f0", anchor=tk.W)
        self._status.pack(fill=tk.X)

        paned = tk.PanedWindow(middle, orient=tk.VERTICAL, sashrelief=tk.RAISED, bg="#f1f5f9")
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        table_fr = tk.Frame(paned, bg="#f1f5f9")
        cols = ("date", "country", "region", "confirmed_cases", "deaths", "recovered")
        self.tree = ttk.Treeview(table_fr, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c.replace("_", " ").title())
            self.tree.column(c, width=100 if c in ("date", "country", "region") else 90)
        scroll = ttk.Scrollbar(table_fr, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        paned.add(table_fr, minsize=160)

        form = tk.LabelFrame(paned, text="Record", font=("Segoe UI", 10, "bold"))
        paned.add(form, minsize=200)
        self._fields: dict[str, tk.Entry] = {}
        field_rows = [
            ("date", "Date (YYYY-MM-DD)"),
            ("country", "Country"),
            ("region", "Region"),
            ("confirmed_cases", "Confirmed"),
            ("new_cases", "New cases"),
            ("deaths", "Deaths"),
            ("new_deaths", "New deaths"),
            ("recovered", "Recovered"),
            ("active_cases", "Active"),
            ("tests_conducted", "Tests"),
            ("hospitalized", "Hospitalized"),
            ("critical", "Critical"),
            ("population", "Population"),
        ]
        for idx, (name, lab) in enumerate(field_rows):
            fr = tk.Frame(form, bg="white")
            fr.grid(row=idx // 4, column=idx % 4, padx=4, pady=4, sticky="ew")
            tk.Label(fr, text=lab, font=("Segoe UI", 8), bg="white").pack(anchor=tk.W)
            e = tk.Entry(fr, width=14)
            e.pack()
            self._fields[name] = e
        for col in range(4):
            form.columnconfigure(col, weight=1)

        btn_fr = tk.Frame(form, bg="white")
        btn_fr.grid(row=4, column=0, columnspan=4, pady=8)
        tk.Button(btn_fr, text="Save", command=self._save).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_fr, text="Clear / New", command=self._clear).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_fr, text="Delete selected", command=self._delete).pack(side=tk.LEFT, padx=4)

        self._reload_table()
        self._clear()

    def _reload_table(self) -> None:
        for i in self.tree.get_children():
            self.tree.delete(i)
        df = self.dm.dataframe().sort_values("date", ascending=False).head(100)
        for _, row in df.iterrows():
            self.tree.insert(
                "",
                tk.END,
                iid=str(row["id"]),
                values=(
                    row["date"],
                    row["country"],
                    row["region"],
                    int(row["confirmed_cases"]),
                    int(row["deaths"]),
                    int(row["recovered"]),
                ),
            )

    def _on_select(self, _: object) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        rid = sel[0]
        df = self.dm.dataframe()
        sub = df[df["id"].astype(str) == rid]
        if sub.empty:
            return
        row = sub.iloc[0]
        self._editing_id = str(row["id"])
        for k in self._fields:
            v = row.get(k, "")
            self._fields[k].delete(0, tk.END)
            self._fields[k].insert(0, str(v))
        self._status.config(text=f"Editing {self._editing_id[:8]}…")

    def _read_form(self) -> dict:
        out: dict = {}
        for k, e in self._fields.items():
            raw = e.get().strip()
            if k in ("date", "country", "region"):
                out[k] = raw
            else:
                try:
                    out[k] = int(float(raw)) if raw else 0
                except ValueError:
                    out[k] = 0
        return out

    def _clear(self) -> None:
        self._editing_id = None
        defaults = {
            "date": "",
            "country": "",
            "region": "",
            "confirmed_cases": 0,
            "new_cases": 0,
            "deaths": 0,
            "new_deaths": 0,
            "recovered": 0,
            "active_cases": 0,
            "tests_conducted": 0,
            "hospitalized": 0,
            "critical": 0,
            "population": 0,
        }
        import datetime

        defaults["date"] = datetime.date.today().isoformat()
        for k, v in defaults.items():
            self._fields[k].delete(0, tk.END)
            self._fields[k].insert(0, str(v))
        self._status.config(text="New record")

    def _save(self) -> None:
        data = self._read_form()
        if not data.get("country") or not data.get("region"):
            messagebox.showwarning("Validation", "Country and region are required.")
            return
        if self._editing_id:
            self.dm.update_row(self._editing_id, data)
            self._status.config(text="Record updated.")
        else:
            self.dm.add_row(data)
            self._status.config(text="Record added.")
        self._reload_table()
        self._on_change()

    def _delete(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Delete", "Select a row first.")
            return
        if not messagebox.askyesno("Confirm", "Delete this record?"):
            return
        self.dm.delete_row(sel[0])
        self._clear()
        self._reload_table()
        self._on_change()


class VisualizationWindow(tk.Toplevel, LayeredWindowMixin):
    """Matplotlib figures (Barnes reference: global trends, waves, heatmap, CFR, scatter)."""

    def __init__(self, master: tk.Tk, dm: DataManager) -> None:
        super().__init__(master)
        self.title("Visualization — Matplotlib (Barnes-style)")
        self.geometry("1120x780")
        self.dm = dm

        _, middle, bottom = self._build_layers(
            self,
            "Global trend visualisation & wave analysis",
            "Figures use dark theme colours (#0D1117) · 7-day rolling averages · NumPy/Pandas aggregates",
        )

        self._status = tk.Label(bottom, text="", font=("Consolas", 9), fg=PALETTE["muted"], bg=PALETTE["surface"], anchor=tk.W)
        self._status.pack(fill=tk.X)

        df = self.dm.dataframe()
        if len(df) == 0:
            tk.Label(middle, text="No data — load CSV or synthetic dataset from Data menu.", fg=PALETTE["muted"], bg=PALETTE["bg"]).pack(expand=True)
            self._status.config(text="Empty database.")
            return

        stats = compute_barnes_statistics(df)
        nb = ttk.Notebook(middle)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        style = ttk.Style()
        try:
            style.theme_use("clam")
            style.configure("TNotebook", background=PALETTE["bg"], borderwidth=0)
            style.configure("TNotebook.Tab", background=PALETTE["surface"], foreground=PALETTE["text"], padding=[12, 6])
            style.map("TNotebook.Tab", background=[("selected", PALETTE["border"])])
        except tk.TclError:
            pass

        tabs: list[tuple[str, object]] = [
            ("02 · Global trends", figure_global_trends(df)),
            ("03 · Waves (USA/IN/BR)", figure_wave_panel(df)),
            ("04 · Country & CFR", figure_country_bars(stats)),
            ("05 · Monthly heatmap", figure_heatmap_monthly(df)),
            ("06 · Burden scatter", figure_scatter_burden(df, stats)),
        ]
        for title, fig in tabs:
            tab = tk.Frame(nb, bg=PALETTE["bg"])
            nb.add(tab, text=title)
            canvas = embed_figure(tab, fig)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._status.config(text=f"Matplotlib · {len(df):,} rows · {stats['global'].get('date_range', '')}")


class StatisticsWindow(tk.Toplevel, LayeredWindowMixin):
    """Barnes-style per-country blocks + series metric table."""

    def __init__(self, master: tk.Tk, dm: DataManager) -> None:
        super().__init__(master)
        self.title("Statistics — NumPy / Pandas (Barnes-style)")
        self.geometry("920x700")
        self.dm = dm
        self._metric: MetricKind = "confirmed_cases"

        top, middle, bottom = self._build_layers(
            self,
            "Statistical analysis",
            "Section 07 style — mean/median/std daily incidence, CFR, recovery, peak daily",
        )

        tb = tk.Frame(top, bg=PALETTE["surface"])
        tb.pack(fill=tk.X)
        for label, val in [("Cases", "confirmed_cases"), ("Deaths", "deaths"), ("Recovered", "recovered")]:
            tk.Button(
                tb,
                text=label,
                command=lambda v=val: self._set_metric(v),
                font=("Segoe UI", 9),
                bg=PALETTE["border"],
                fg=PALETTE["text"],
                relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(
            middle,
            text="07 — NumPy statistical computation (latest cumulative + daily distributional stats)",
            font=("Consolas", 9),
            fg=PALETTE["accent1"],
            bg=PALETTE["bg"],
        ).pack(anchor=tk.W, padx=10, pady=(6, 2))

        self._barnes = tk.Text(
            middle,
            height=14,
            font=("Consolas", 9),
            wrap=tk.WORD,
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["text"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        self._barnes.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        tk.Label(
            middle,
            text="Series metric (all rows) — top countries by sum of selected column",
            font=("Consolas", 9),
            fg=PALETTE["muted"],
            bg=PALETTE["bg"],
        ).pack(anchor=tk.W, padx=10)

        self._summary = tk.Text(
            middle,
            height=6,
            font=("Consolas", 9),
            wrap=tk.WORD,
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["text"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        self._summary.pack(fill=tk.X, padx=10, pady=4)

        cols = ("country", "mean", "median", "total", "min", "max")
        self.tree = ttk.Treeview(middle, columns=cols, show="headings", height=8)
        heads = ["Country", "Mean", "Median", "Total", "Min", "Max"]
        for c, h in zip(cols, heads):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=110)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self._status = tk.Label(bottom, text="", font=("Consolas", 9), fg=PALETTE["muted"], bg=PALETTE["surface"], anchor=tk.W)
        self._status.pack(fill=tk.X)

        self._compute()

    def _set_metric(self, m: MetricKind) -> None:
        self._metric = m
        self._compute()

    def _compute(self) -> None:
        df = self.dm.dataframe().sort_values("date")
        col = self._metric

        self._barnes.delete("1.0", tk.END)
        if len(df):
            st = compute_barnes_statistics(df)
            g = st["global"]
            self._barnes.insert(
                tk.END,
                f"GLOBAL  ·  {g.get('date_range', '')}\n"
                f"Confirmed {fmt_compact(g['total_confirmed'])}  |  Deaths {fmt_compact(g['total_deaths'])}  |  "
                f"CFR {g['global_cfr']}%  |  Recovered {fmt_compact(g['total_recovered'])}  |  "
                f"Recovery {g['global_recovery']}%  |  Active {fmt_compact(g['total_active'])}\n\n",
            )
            for c, d in sorted(st["countries"].items(), key=lambda x: x[1]["confirmed"], reverse=True):
                self._barnes.insert(
                    tk.END,
                    f"▸ {c}\n"
                    f"   Mean daily cases {d['mean_daily']:,}  ·  Median {d['median_daily']:,}  ·  Std {d['std_daily']:,}\n"
                    f"   Peak daily {d['peak_daily']:,}  ·  CFR {d['cfr']}%  ·  Recovery {d['recovery_rate']}%  ·  "
                    f"Cases/M {d['cases_per_1m']}  ·  Deaths/M {d['deaths_per_1m']}\n\n",
                )
        else:
            self._barnes.insert(tk.END, "No data.\n")

        if len(df) == 0:
            self._summary.delete("1.0", tk.END)
            self._status.config(text="No rows.")
            return

        vals = df[col].astype(float).values
        mean = np_mean(vals)
        med = np_median(vals)
        std = np_std(vals)
        sm = np_sum(vals)
        mn = _np_min(vals)
        mx = _np_max(vals)

        recent = df.tail(7)
        prev = df.tail(14).head(7)
        recent_sum = float(recent[col].sum()) if len(recent) else 0.0
        previous_sum = float(prev[col].sum()) if len(prev) == 7 else 0.0
        gr = growth_rate(recent_sum, previous_sum) if len(df) >= 2 else 0.0
        trend = "up" if gr > 0 else "down" if gr < 0 else "neutral"

        s_series = df[col].astype(float)
        win = min(7, len(s_series))
        roll = s_series.rolling(window=win, min_periods=1).mean()
        last_ma = float(roll.iloc[-1]) if len(roll) else 0.0
        cv_pct = (std / mean * 100.0) if mean and mean != 0 else 0.0

        self._summary.delete("1.0", tk.END)
        self._summary.insert(
            tk.END,
            f"Metric column: {col}\n"
            f"Sum: {fmt_num(sm)}  |  Mean: {mean:.2f}  |  Median: {med:.2f}  |  Std: {std:.2f}\n"
            f"Min: {fmt_num(mn)}  |  Max: {fmt_num(mx)}  |  Range: {fmt_num(mx - mn)}\n"
            f"CV: {cv_pct:.2f}%  |  Rolling mean (w={win}): {last_ma:.2f}  |  7d vs prev 7d growth: {gr:.2f}% ({trend})\n",
        )

        for i in self.tree.get_children():
            self.tree.delete(i)
        rows: list[tuple[float, tuple[str, str, str, str, str, str]]] = []
        for country, g in df.groupby("country"):
            v = g[col].astype(float).values
            total = np_sum(v)
            rows.append(
                (
                    total,
                    (
                        country,
                        f"{np_mean(v):.2f}",
                        f"{np_median(v):.2f}",
                        fmt_num(total),
                        fmt_num(_np_min(v)),
                        fmt_num(_np_max(v)),
                    ),
                )
            )
        for _, vals in sorted(rows, key=lambda x: x[0], reverse=True)[:10]:
            self.tree.insert("", tk.END, values=vals)

        self._status.config(text=f"{len(df):,} rows · top 10 countries by Σ{col}")


class SearchFilterWindow(tk.Toplevel, LayeredWindowMixin):
    def __init__(self, master: tk.Tk, dm: DataManager) -> None:
        super().__init__(master)
        self.title("Search & Filter — COVID-19")
        self.geometry("960x560")
        self.dm = dm

        _, middle, bottom = self._build_layers(self, "Search & Filter", "Query and narrow case records")

        filt = tk.Frame(middle, bg="#f1f5f9")
        filt.pack(fill=tk.X, padx=12, pady=8)
        self._search = tk.Entry(filt, width=40)
        self._search.pack(side=tk.LEFT, padx=4)
        self._search.bind("<KeyRelease>", lambda e: self._apply())

        fields = [
            ("country", "Country"),
            ("region", "Region"),
            ("date_from", "Date from"),
            ("date_to", "Date to"),
            ("min_cases", "Min cases"),
            ("max_cases", "Max cases"),
            ("min_deaths", "Min deaths"),
            ("max_deaths", "Max deaths"),
        ]
        self._filters: dict[str, tk.Entry] = {}
        grid = tk.Frame(filt, bg="#f1f5f9")
        grid.pack(fill=tk.X, pady=8)
        for i, (name, lab) in enumerate(fields):
            r, c = divmod(i, 4)
            fr = tk.Frame(grid, bg="#f1f5f9")
            fr.grid(row=r, column=c, padx=4, pady=2)
            tk.Label(fr, text=lab, font=("Segoe UI", 8), bg="#f1f5f9").pack(anchor=tk.W)
            e = tk.Entry(fr, width=16)
            e.pack()
            e.bind("<KeyRelease>", lambda ev: self._apply())
            self._filters[name] = e

        tk.Button(filt, text="Clear filters", command=self._clear).pack(side=tk.RIGHT, padx=8)

        cols = ("date", "country", "region", "confirmed_cases", "deaths", "recovered")
        self.tree = ttk.Treeview(middle, columns=cols, show="headings", height=18)
        for c in cols:
            self.tree.heading(c, text=c.replace("_", " ").title())
            self.tree.column(c, width=110)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self._status = tk.Label(bottom, text="", font=("Segoe UI", 9), fg="#334155", bg="#e2e8f0", anchor=tk.W)
        self._status.pack(fill=tk.X)

        self._apply()

    def _clear(self) -> None:
        self._search.delete(0, tk.END)
        for e in self._filters.values():
            e.delete(0, tk.END)
        self._apply()

    def _apply(self, *_args: object) -> None:
        df = self.dm.dataframe()
        term = self._search.get().strip().lower()
        if term:
            df = df[
                df["country"].str.lower().str.contains(term, na=False)
                | df["region"].str.lower().str.contains(term, na=False)
            ]
        cty = self._filters["country"].get().strip()
        if cty:
            df = df[df["country"].str.lower().str.contains(cty.lower(), na=False)]
        reg = self._filters["region"].get().strip()
        if reg:
            df = df[df["region"].str.lower().str.contains(reg.lower(), na=False)]
        df0 = self._filters["date_from"].get().strip()
        df1 = self._filters["date_to"].get().strip()
        if df0:
            df = df[df["date"].astype(str) >= df0]
        if df1:
            df = df[df["date"].astype(str) <= df1]

        def _int_entry(key: str) -> int | None:
            s = self._filters[key].get().strip()
            if not s:
                return None
            try:
                return int(s)
            except ValueError:
                return None

        mn_c = _int_entry("min_cases")
        mx_c = _int_entry("max_cases")
        mn_d = _int_entry("min_deaths")
        mx_d = _int_entry("max_deaths")
        if mn_c is not None:
            df = df[df["confirmed_cases"] >= mn_c]
        if mx_c is not None:
            df = df[df["confirmed_cases"] <= mx_c]
        if mn_d is not None:
            df = df[df["deaths"] >= mn_d]
        if mx_d is not None:
            df = df[df["deaths"] <= mx_d]

        for i in self.tree.get_children():
            self.tree.delete(i)
        for _, row in df.sort_values("date", ascending=False).iterrows():
            self.tree.insert(
                "",
                tk.END,
                values=(
                    row["date"],
                    row["country"],
                    row["region"],
                    int(row["confirmed_cases"]),
                    int(row["deaths"]),
                    int(row["recovered"]),
                ),
            )
        self._status.config(text=f"{len(df)} row(s) match.")


class DatabasePipelineWindow(tk.Toplevel, LayeredWindowMixin):
    """Data ingestion, preprocessing on DB, and SQL SELECT retrieval (SQLite)."""

    def __init__(self, master: tk.Tk, dm: DataManager, on_change: Callable[[], None]) -> None:
        super().__init__(master)
        self.title("Data pipeline & database — COVID-19")
        self.geometry("920x640")
        self.dm = dm
        self._on_change = on_change

        _, middle, bottom = self._build_layers(
            self,
            "Ingestion, preprocessing & query",
            "Import public CSVs, clean data, persist in SQLite, run SELECT analytics",
        )

        self._status = tk.Label(bottom, text="", font=("Consolas", 9), fg=PALETTE["muted"], bg=PALETTE["surface"], anchor=tk.W)
        self._status.pack(fill=tk.X)
        self._status.config(text=f"Database: {self.dm.db_path}")

        ingest = tk.LabelFrame(
            middle,
            text="1. Ingest CSV (data ingestion)",
            font=("Consolas", 10, "bold"),
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        ingest.pack(fill=tk.X, padx=12, pady=8)
        self._append_var = tk.BooleanVar(value=True)
        tk.Radiobutton(
            ingest,
            text="Append rows",
            variable=self._append_var,
            value=True,
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            selectcolor=PALETTE["border"],
        ).grid(row=0, column=0, sticky=tk.W)
        tk.Radiobutton(
            ingest,
            text="Replace all cases",
            variable=self._append_var,
            value=False,
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            selectcolor=PALETTE["border"],
        ).grid(row=0, column=1, sticky=tk.W)
        tk.Button(ingest, text="Choose CSV and import…", command=self._do_import).grid(row=1, column=0, columnspan=2, pady=6, sticky=tk.W)

        pre = tk.LabelFrame(
            middle,
            text="2. Preprocessing (Pandas) on entire database",
            font=("Consolas", 10, "bold"),
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        pre.pack(fill=tk.X, padx=12, pady=8)
        self._opt_strip = tk.BooleanVar(value=True)
        self._opt_coerce = tk.BooleanVar(value=True)
        self._opt_dates = tk.BooleanVar(value=True)
        self._opt_dedupe = tk.BooleanVar(value=True)
        tk.Checkbutton(pre, text="Strip strings", variable=self._opt_strip, bg=PALETTE["surface"], fg=PALETTE["text"]).grid(row=0, column=0, sticky=tk.W)
        tk.Checkbutton(pre, text="Coerce / fill numerics", variable=self._opt_coerce, bg=PALETTE["surface"], fg=PALETTE["text"]).grid(row=0, column=1, sticky=tk.W)
        tk.Checkbutton(pre, text="Normalize dates", variable=self._opt_dates, bg=PALETTE["surface"], fg=PALETTE["text"]).grid(row=0, column=2, sticky=tk.W)
        tk.Checkbutton(pre, text="Drop duplicates (date, country, region)", variable=self._opt_dedupe, bg=PALETTE["surface"], fg=PALETTE["text"]).grid(
            row=1, column=0, columnspan=2, sticky=tk.W
        )
        tk.Button(pre, text="Run preprocessing on all stored rows", command=self._do_preprocess).grid(row=2, column=0, pady=6, sticky=tk.W)

        sql_fr = tk.LabelFrame(
            middle,
            text="3. Query-based retrieval (read-only SELECT) — section 08 style",
            font=("Consolas", 10, "bold"),
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        sql_fr.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        hint = "Example: aggregate new cases by country; use MAX(date) filters for latest snapshots."
        tk.Label(sql_fr, text=hint, font=("Segoe UI", 8), fg=PALETTE["muted"], bg=PALETTE["surface"], wraplength=860).pack(anchor=tk.W)
        preset_fr = tk.Frame(sql_fr, bg=PALETTE["surface"])
        preset_fr.pack(fill=tk.X, pady=4)
        presets = [
            ("Σ new cases by country", "SELECT country, SUM(new_cases) AS sum_new, SUM(new_deaths) AS sum_d FROM covid_cases GROUP BY country ORDER BY sum_new DESC LIMIT 12"),
            ("Latest cumulative (max per country)", "SELECT country, MAX(confirmed_cases) AS cum_c, MAX(deaths) AS cum_d, ROUND(100.0*MAX(deaths)/NULLIF(MAX(confirmed_cases),0),3) AS cfr_pct FROM covid_cases GROUP BY country ORDER BY cum_c DESC LIMIT 12"),
            ("USA last 30 daily rows", "SELECT date, new_cases, new_deaths, confirmed_cases FROM covid_cases WHERE country='USA' ORDER BY date DESC LIMIT 30"),
        ]
        for lab, q in presets:
            tk.Button(
                preset_fr,
                text=lab,
                font=("Consolas", 8),
                bg=PALETTE["border"],
                fg=PALETTE["text"],
                relief=tk.FLAT,
                command=lambda sql=q: (self._sql_text.delete("1.0", tk.END), self._sql_text.insert("1.0", sql)),
            ).pack(side=tk.LEFT, padx=4, pady=2)
        self._sql_text = tk.Text(sql_fr, height=4, font=("Consolas", 10), bg=PALETTE["bg"], fg=PALETTE["text"], insertbackground=PALETTE["text"])
        self._sql_text.pack(fill=tk.X, pady=4)
        self._sql_text.insert("1.0", "SELECT * FROM covid_cases LIMIT 50")
        tk.Button(sql_fr, text="Run query", command=self._do_sql).pack(anchor=tk.W, pady=4)
        out_wrap = tk.Frame(sql_fr)
        out_wrap.pack(fill=tk.BOTH, expand=True)
        sy = ttk.Scrollbar(out_wrap, orient=tk.VERTICAL)
        sx = ttk.Scrollbar(out_wrap, orient=tk.HORIZONTAL)
        self._sql_out = tk.Text(out_wrap, height=12, font=("Consolas", 9), wrap=tk.NONE, yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.config(command=self._sql_out.yview)
        sx.config(command=self._sql_out.xview)
        self._sql_out.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        out_wrap.rowconfigure(0, weight=1)
        out_wrap.columnconfigure(0, weight=1)

    def _options_from_ui(self) -> PreprocessOptions:
        return PreprocessOptions(
            strip_strings=self._opt_strip.get(),
            coerce_numeric=self._opt_coerce.get(),
            parse_dates=self._opt_dates.get(),
            drop_duplicates=self._opt_dedupe.get(),
        )

    def _do_import(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            n, msgs = self.dm.ingest_csv(path, append=self._append_var.get(), preprocess_options=self._options_from_ui())
            self._on_change()
            self._status.config(text=f"Imported {n} row(s). DB: {self.dm.db_path}")
            messagebox.showinfo("Ingestion", f"Rows written: {n}\n\n" + "\n".join(msgs))
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def _do_preprocess(self) -> None:
        if not messagebox.askyesno(
            "Preprocess",
            "This rewrites all case rows after cleaning (new row IDs). Continue?",
        ):
            return
        try:
            n, msgs = self.dm.apply_preprocessing_to_database(self._options_from_ui())
            self._on_change()
            self._status.config(text=f"Preprocessed {n} row(s).")
            messagebox.showinfo("Preprocessing", f"Rows after cleaning: {n}\n\n" + "\n".join(msgs))
        except Exception as e:
            messagebox.showerror("Preprocessing failed", str(e))

    def _do_sql(self) -> None:
        sql = self._sql_text.get("1.0", tk.END).strip()
        self._sql_out.delete("1.0", tk.END)
        try:
            res = self.dm.run_select_query(sql)
            self._sql_out.insert(tk.END, res.to_string(index=False))
            self._status.config(text=f"Query returned {len(res)} row(s).")
        except Exception as e:
            self._sql_out.insert(tk.END, str(e))
            messagebox.showerror("Query error", str(e))


class ReportsWindow(tk.Toplevel, LayeredWindowMixin):
    def __init__(self, master: tk.Tk, dm: DataManager) -> None:
        super().__init__(master)
        self.title("Reports — COVID-19")
        self.geometry("820x580")
        self.dm = dm

        top, middle, bottom = self._build_layers(self, "Analysis reports", "Generate summary reports for a date range")

        gen = tk.LabelFrame(middle, text="Generate report", font=("Segoe UI", 10, "bold"), bg="#f1f5f9")
        gen.pack(fill=tk.X, padx=12, pady=8)
        self._name = tk.Entry(gen, width=40)
        self._name.grid(row=0, column=1, padx=4, pady=4)
        tk.Label(gen, text="Report name", bg="#f1f5f9").grid(row=0, column=0, sticky=tk.W)
        self._d0 = tk.Entry(gen, width=14)
        self._d0.grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        tk.Label(gen, text="Date from (YYYY-MM-DD)", bg="#f1f5f9").grid(row=1, column=0, sticky=tk.W)
        self._d1 = tk.Entry(gen, width=14)
        self._d1.grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)
        tk.Label(gen, text="Date to", bg="#f1f5f9").grid(row=2, column=0, sticky=tk.W)
        tk.Button(gen, text="Generate", command=self._generate).grid(row=3, column=1, sticky=tk.W, pady=8)

        list_fr = tk.LabelFrame(middle, text="Saved reports", font=("Segoe UI", 10, "bold"))
        list_fr.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self._list = tk.Listbox(list_fr, font=("Segoe UI", 10), height=10)
        self._list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        sc = ttk.Scrollbar(list_fr, command=self._list.yview)
        sc.pack(side=tk.RIGHT, fill=tk.Y)
        self._list.config(yscrollcommand=sc.set)

        detail = tk.Text(list_fr, height=12, width=50, font=("Consolas", 9), wrap=tk.WORD)
        detail.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._detail = detail

        btn_fr = tk.Frame(middle, bg="#f1f5f9")
        btn_fr.pack(fill=tk.X)
        tk.Button(btn_fr, text="Show details", command=lambda: self._show_detail()).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_fr, text="Delete selected", command=self._delete).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_fr, text="Export JSON…", command=self._export).pack(side=tk.LEFT, padx=8)

        self._status = tk.Label(bottom, text="", font=("Segoe UI", 9), fg="#334155", bg="#e2e8f0", anchor=tk.W)
        self._status.pack(fill=tk.X)

        self._reload_list()

    def _reload_list(self) -> None:
        self._list.delete(0, tk.END)
        self._reports = self.dm.load_reports()
        for r in self._reports:
            self._list.insert(tk.END, f"{r.get('report_name', '')} — {r.get('date_from')} … {r.get('date_to')}")

    def _generate(self) -> None:
        name = self._name.get().strip()
        d0 = self._d0.get().strip()
        d1 = self._d1.get().strip()
        if not name or not d0 or not d1:
            messagebox.showwarning("Reports", "Name, date from, and date to are required.")
            return
        df = self.dm.dataframe()
        sub = df[(df["date"].astype(str) >= d0) & (df["date"].astype(str) <= d1)]
        data = {
            "totalRecords": int(len(sub)),
            "totalCases": float(np_sum(sub["confirmed_cases"].values)),
            "totalDeaths": float(np_sum(sub["deaths"].values)),
            "totalRecovered": float(np_sum(sub["recovered"].values)),
            "averageCases": np_mean(sub["confirmed_cases"].values) if len(sub) else 0.0,
            "averageDeaths": np_mean(sub["deaths"].values) if len(sub) else 0.0,
            "countries": sub["country"].unique().tolist(),
            "regions": sub["region"].unique().tolist(),
            "dateRange": {"from": d0, "to": d1},
        }
        self.dm.add_report(
            {
                "report_name": name,
                "report_type": "summary",
                "date_from": d0,
                "date_to": d1,
                "regions": [],
                "data": data,
            }
        )
        self._name.delete(0, tk.END)
        self._reload_list()
        self._status.config(text="Report saved.")

    def _show_detail(self) -> None:
        sel = self._list.curselection()
        if not sel:
            return
        r = self._reports[sel[0]]
        self._detail.delete("1.0", tk.END)
        import json

        self._detail.insert(tk.END, json.dumps(r.get("data", {}), indent=2))

    def _delete(self) -> None:
        sel = self._list.curselection()
        if not sel:
            return
        if not messagebox.askyesno("Delete", "Delete this report?"):
            return
        rid = self._reports[sel[0]].get("id")
        if rid:
            self.dm.delete_report(str(rid))
        self._reload_list()
        self._detail.delete("1.0", tk.END)

    def _export(self) -> None:
        sel = self._list.curselection()
        if not sel:
            messagebox.showinfo("Export", "Select a report.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        import json

        Path(path).write_text(json.dumps(self._reports[sel[0]], indent=2), encoding="utf-8")
        self._status.config(text=f"Exported to {path}")


def main() -> None:
    app = CovidMainApp()
    app.mainloop()


if __name__ == "__main__":
    main()
