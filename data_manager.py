"""SQLite database + Pandas/NumPy for COVID-19 analytics. CSV/JSON migration on first run."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "id",
    "date",
    "country",
    "region",
    "confirmed_cases",
    "new_cases",
    "deaths",
    "new_deaths",
    "recovered",
    "active_cases",
    "tests_conducted",
    "hospitalized",
    "critical",
    "population",
]

STRING_COLS = ["date", "country", "region"]
NUMERIC_COLS = [c for c in COLUMNS if c not in ("id",) and c not in STRING_COLS]


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


class DataManager:
    """
    Persistent storage in SQLite; Pandas for frames; NumPy helpers for aggregates.
    Supports ingestion from CSV, preprocessing, and read-only SQL retrieval.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or _default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "covid19.db"
        self.legacy_csv = self.data_dir / "covid_cases.csv"
        self.legacy_reports_json = self.data_dir / "analysis_reports.json"
        self._init_schema()
        self._migrate_legacy_if_needed()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS covid_cases (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    country TEXT NOT NULL,
                    region TEXT NOT NULL,
                    confirmed_cases INTEGER NOT NULL DEFAULT 0,
                    new_cases INTEGER NOT NULL DEFAULT 0,
                    deaths INTEGER NOT NULL DEFAULT 0,
                    new_deaths INTEGER NOT NULL DEFAULT 0,
                    recovered INTEGER NOT NULL DEFAULT 0,
                    active_cases INTEGER NOT NULL DEFAULT 0,
                    tests_conducted INTEGER NOT NULL DEFAULT 0,
                    hospitalized INTEGER NOT NULL DEFAULT 0,
                    critical INTEGER NOT NULL DEFAULT 0,
                    population INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_cases_date ON covid_cases(date);
                CREATE INDEX IF NOT EXISTS idx_cases_country ON covid_cases(country);

                CREATE TABLE IF NOT EXISTS analysis_reports (
                    id TEXT PRIMARY KEY,
                    report_name TEXT NOT NULL,
                    report_type TEXT,
                    date_from TEXT NOT NULL,
                    date_to TEXT NOT NULL,
                    regions_json TEXT,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _migrate_legacy_if_needed(self) -> None:
        with self._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM covid_cases").fetchone()[0]
        if n == 0 and self.legacy_csv.exists():
            df = pd.read_csv(self.legacy_csv)
            from preprocessing import PreprocessOptions, preprocess_covid_dataframe

            pr = preprocess_covid_dataframe(df, PreprocessOptions())
            fixed = pr.dataframe.copy()
            fixed["id"] = [str(uuid.uuid4()) for _ in range(len(fixed))]
            self._replace_cases_table(fixed)
        elif n == 0:
            self._insert_seed_rows()

        with self._connect() as conn:
            nr = conn.execute("SELECT COUNT(*) FROM analysis_reports").fetchone()[0]
        if nr == 0 and self.legacy_reports_json.exists():
            try:
                raw = self.legacy_reports_json.read_text(encoding="utf-8")
                reports = json.loads(raw) if raw.strip() else []
                for r in reports:
                    self._insert_report_row(r)
            except (json.JSONDecodeError, OSError):
                pass

    def _insert_seed_rows(self) -> None:
        rows = [
            (
                str(uuid.uuid4()),
                "2024-01-15",
                "United States",
                "California",
                1200,
                80,
                12,
                1,
                900,
                288,
                5000,
                40,
                8,
                39500000,
            ),
            (
                str(uuid.uuid4()),
                "2024-01-16",
                "United States",
                "Texas",
                950,
                55,
                9,
                0,
                700,
                241,
                4200,
                32,
                5,
                30000000,
            ),
            (
                str(uuid.uuid4()),
                "2024-01-15",
                "Germany",
                "Bavaria",
                600,
                40,
                5,
                0,
                500,
                95,
                3100,
                18,
                3,
                13000000,
            ),
            (
                str(uuid.uuid4()),
                "2024-01-17",
                "Germany",
                "Berlin",
                420,
                28,
                3,
                0,
                360,
                57,
                2100,
                12,
                2,
                3700000,
            ),
            (
                str(uuid.uuid4()),
                "2024-01-14",
                "Japan",
                "Tokyo",
                800,
                50,
                4,
                0,
                720,
                76,
                4500,
                22,
                4,
                14000000,
            ),
        ]
        cols_sql = ", ".join(COLUMNS[1:])
        placeholders = ", ".join(["?"] * (len(COLUMNS) - 1))
        with self._connect() as conn:
            conn.executemany(
                f"INSERT INTO covid_cases (id, {cols_sql}) VALUES (?, {placeholders})",
                rows,
            )

    def _replace_cases_table(self, df: pd.DataFrame) -> None:
        df = df.copy()
        if "id" not in df.columns or df["id"].astype(str).str.strip().eq("").any():
            df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = 0 if c != "id" else ""
        df = df[COLUMNS]
        for k in NUMERIC_COLS:
            df[k] = df[k].astype(np.int64)
        with self._connect() as conn:
            conn.execute("DELETE FROM covid_cases")
            df.to_sql("covid_cases", conn, if_exists="append", index=False)

    def dataframe(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM covid_cases", conn)

    def dashboard_stats(self) -> dict[str, int | float]:
        """KPIs use latest row per country (correct for cumulative time-series, Barnes-style)."""
        df = self.dataframe()
        if len(df) == 0:
            return {
                "total_records": 0,
                "total_cases": 0,
                "total_deaths": 0,
                "total_recovered": 0,
                "total_active": 0,
                "countries": 0,
                "cfr_pct": 0.0,
                "recovery_pct": 0.0,
            }
        latest = df.sort_values("date").groupby("country", as_index=False).last()
        tc = int(latest["confirmed_cases"].sum())
        td = int(latest["deaths"].sum())
        tr = int(latest["recovered"].sum())
        ta = int(latest["active_cases"].sum())
        cfr = (td / tc * 100.0) if tc else 0.0
        rr = (tr / tc * 100.0) if tc else 0.0
        return {
            "total_records": int(len(df)),
            "total_cases": tc,
            "total_deaths": td,
            "total_recovered": tr,
            "total_active": ta,
            "countries": int(latest["country"].nunique()),
            "cfr_pct": round(cfr, 3),
            "recovery_pct": round(rr, 2),
        }

    def add_row(self, row: dict) -> None:
        new_id = str(uuid.uuid4())
        r = {k: row.get(k, 0) for k in COLUMNS}
        r["id"] = new_id
        for k in NUMERIC_COLS:
            r[k] = int(r[k]) if r[k] is not None else 0
        cols = ", ".join(COLUMNS)
        placeholders = ", ".join(["?"] * len(COLUMNS))
        vals = [r[c] for c in COLUMNS]
        with self._connect() as conn:
            conn.execute(f"INSERT INTO covid_cases ({cols}) VALUES ({placeholders})", vals)

    def update_row(self, row_id: str, row: dict) -> None:
        df = self.dataframe()
        sub = df[df["id"].astype(str) == row_id]
        if sub.empty:
            return
        cur = sub.iloc[0].to_dict()
        for k in COLUMNS:
            if k != "id" and k in row:
                cur[k] = row[k]
        sets = ", ".join([f"{k} = ?" for k in COLUMNS if k != "id"])
        vals = [int(cur[k]) if k in NUMERIC_COLS else cur[k] for k in COLUMNS if k != "id"]
        vals.append(row_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE covid_cases SET {sets} WHERE id = ?", vals)

    def delete_row(self, row_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM covid_cases WHERE id = ?", (row_id,))

    def ingest_csv(
        self,
        path: str | Path,
        *,
        append: bool = True,
        preprocess_options: "PreprocessOptions | None" = None,
    ) -> tuple[int, list[str]]:
        """
        Data ingestion: load a public CSV, preprocess, persist to SQLite.
        Returns (rows_inserted, log_messages).
        """
        from preprocessing import PreprocessOptions, preprocess_covid_dataframe

        opts = preprocess_options or PreprocessOptions()
        raw = pd.read_csv(path)
        pr = preprocess_covid_dataframe(raw, opts)
        df = pr.dataframe.copy()
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df = df[COLUMNS]
        with self._connect() as conn:
            if not append:
                conn.execute("DELETE FROM covid_cases")
            df.to_sql("covid_cases", conn, if_exists="append", index=False)
        return len(df), pr.messages

    def seed_synthetic_full(self) -> int:
        """Replace all cases with full Barnes-style synthetic multi-country time series (~12k rows)."""
        from synthetic_data import generate_synthetic_cases_dataframe

        df = generate_synthetic_cases_dataframe()
        self._replace_cases_table(df)
        return len(df)

    def apply_preprocessing_to_database(self, options: "PreprocessOptions | None" = None) -> tuple[int, list[str]]:
        """Reload all cases, run preprocessing, replace table (data cleaning pipeline)."""
        from preprocessing import PreprocessOptions, preprocess_covid_dataframe

        df = self.dataframe()
        if df.empty:
            return 0, ["No rows to preprocess."]
        opts = options or PreprocessOptions()
        df = df.drop(columns=["id"], errors="ignore")
        pr = preprocess_covid_dataframe(df, opts)
        out = pr.dataframe
        out["id"] = [str(uuid.uuid4()) for _ in range(len(out))]
        self._replace_cases_table(out)
        return len(out), pr.messages

    _FORBIDDEN_SQL = re.compile(
        r"\b(DROP|DELETE|INSERT|UPDATE|ATTACH|DETACH|PRAGMA|REPLACE|CREATE|ALTER|TRIGGER|VACUUM)\b",
        re.IGNORECASE,
    )

    def run_select_query(self, sql: str) -> pd.DataFrame:
        """
        Query-based retrieval: run a single SELECT against the database (read-only analytics).
        """
        s = sql.strip().rstrip(";")
        if not s.upper().startswith("SELECT"):
            raise ValueError("Only a single SELECT statement is allowed.")
        if ";" in s:
            raise ValueError("Multiple statements are not allowed.")
        if self._FORBIDDEN_SQL.search(s):
            raise ValueError("Query contains forbidden keywords.")
        with self._connect() as conn:
            return pd.read_sql_query(s, conn)

    def load_reports(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, report_name, report_type, date_from, date_to, regions_json, data_json, created_at FROM analysis_reports ORDER BY created_at DESC"
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            regions = json.loads(r[5]) if r[5] else []
            data = json.loads(r[6]) if r[6] else {}
            out.append(
                {
                    "id": r[0],
                    "report_name": r[1],
                    "report_type": r[2],
                    "date_from": r[3],
                    "date_to": r[4],
                    "regions": regions,
                    "data": data,
                    "created_at": r[7],
                }
            )
        return out

    def _insert_report_row(self, report: dict) -> None:
        rid = report.get("id") or str(uuid.uuid4())
        regions = json.dumps(report.get("regions") or [])
        data_json = json.dumps(report.get("data") or {})
        created = report.get("created_at") or pd.Timestamp.utcnow().isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_reports
                (id, report_name, report_type, date_from, date_to, regions_json, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    report.get("report_name", ""),
                    report.get("report_type", "summary"),
                    report.get("date_from", ""),
                    report.get("date_to", ""),
                    regions,
                    data_json,
                    created,
                ),
            )

    def add_report(self, report: dict) -> None:
        report = dict(report)
        report["id"] = str(uuid.uuid4())
        report["created_at"] = pd.Timestamp.utcnow().isoformat() + "Z"
        self._insert_report_row(report)

    def delete_report(self, report_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM analysis_reports WHERE id = ?", (report_id,))

    def export_cases_csv(self, path: str | Path) -> None:
        self.dataframe().to_csv(path, index=False)


def np_mean(a: np.ndarray | list) -> float:
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def np_median(a: np.ndarray | list) -> float:
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def np_std(a: np.ndarray | list) -> float:
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr, ddof=0))


def np_sum(a: np.ndarray | list) -> float:
    arr = np.asarray(a, dtype=float)
    return float(np.sum(arr))


def growth_rate(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous * 100.0
