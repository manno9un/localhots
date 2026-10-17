"""localhots web server.

Runs on IBM LinuxONE inside the pinned s390x JupyterLab image (it already ships numpy,
pandas and scikit-learn). Loads the retained model from the read-only project mount,
scores the whole 2022-2024 replay period at startup, and serves the website plus a small
JSON API. Standard library HTTP only, so there is nothing extra to install on s390x.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from datetime import date
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PROJECT = Path(os.environ.get("WILDFIRE_DIR", "/work"))
STATIC = Path(__file__).resolve().parent / "static"
PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "0.0.0.0")
REVIEW_K = 3  # cells a crew checks per day

sys.path.insert(0, str(PROJECT / "model"))
from predict import ANOMALIES, BASE_FEATURES, infer  # noqa: E402  (teammate's retained runtime)

# Grid definition from REPORT.md section 2.2.
LAT0, LON0, STEP = 38.5, -123.0, 0.5

FEATURE_LABELS = {
    "latitude": "Latitude", "longitude": "Longitude", "season_sin": "Time of year", "season_cos": "Time of year",
    "tmax_lag1": "Yesterday's high", "tmax_mean7": "7-day avg high", "tmax_max7": "7-day peak high",
    "rhmin_lag1": "Yesterday's humidity", "rhmin_mean7": "7-day avg humidity", "rhmin_min7": "7-day lowest humidity",
    "windmax_lag1": "Yesterday's wind", "windmax_max3": "3-day peak wind", "rain_lag1": "Yesterday's rain",
    "rain_sum7": "Rain, last 7 days", "rain_sum14": "Rain, last 14 days", "rain_sum30": "Rain, last 30 days",
    "dry_days": "Dry days in a row", "cell_code": "Which area", "month": "Month", "weekday": "Day of week",
    "historical_prior": "Past fire rate", "tmax_lag1_anomaly": "Heat vs normal", "tmax_mean7_anomaly": "Week heat vs normal",
    "rhmin_lag1_anomaly": "Humidity vs normal", "windmax_lag1_anomaly": "Wind vs normal",
    "rain_sum7_anomaly": "Week rain vs normal", "rain_sum30_anomaly": "Month rain vs normal",
}


def log(msg: str) -> None:
    print(f"[wildfire] {msg}", flush=True)


def build_features(bundle: dict, rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Same 27-column assembly as predict.infer, exposed so we can explain and audit scores."""
    state = bundle["history"]
    months = rows.target_date.dt.month.to_numpy()
    keys = [f"{c}:{m}" for c, m in zip(rows.cell_id, months)]
    baseline = np.array([state["baseline"]["rates"].get(k, state["baseline"]["global_rate"]) for k in keys])
    x = rows[BASE_FEATURES].copy()
    x["cell_code"] = rows.cell_id.map(state["cell_codes"]).astype(int)
    x["month"] = months
    x["weekday"] = rows.target_date.dt.dayofweek.to_numpy()
    x["historical_prior"] = baseline
    for name in ANOMALIES:
        stats = np.array([state["climate"][k][name] for k in keys])
        x[name + "_anomaly"] = ((rows[name].to_numpy() - stats[:, 0]) / stats[:, 1]).clip(-12, 12)
    return x[bundle["features"]], baseline


def reasons_for(r) -> list[str]:
    """Plain-language drivers from the climate anomalies (z-scores vs. that area's normal for the month)."""
    out = []
    if r.tmax_lag1_anomaly >= 1 or r.tmax_mean7_anomaly >= 1:
        out.append("Hotter than normal for this time of year")
    if r.rhmin_lag1_anomaly <= -1:
        out.append("Drier air than normal")
    if r.windmax_lag1_anomaly >= 1:
        out.append("Windier than normal")
    if r.dry_days >= 30:
        out.append("No real rain for 30+ days")
    elif r.dry_days >= 14:
        out.append(f"{int(r.dry_days)} dry days in a row")
    if r.rain_sum30_anomaly <= -0.75 and r.dry_days < 14:
        out.append("Less rain than normal this month")
    if r.historical_prior >= 0.3:
        out.append("Area often has fires this month")
    if r.rain_sum7 >= 10:
        out.append("Recent rain is lowering risk")
    return out[:3]


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


class Outlook:
    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.bundle = joblib.load(PROJECT / "model/model.joblib")
        manifest = json.loads((PROJECT / "model/manifest.json").read_text())
        data = pd.read_csv(PROJECT / "datasets/model_inputs.csv.gz", float_precision="round_trip", parse_dates=["target_date"])
        cutoff = pd.Timestamp(self.bundle["history"]["fitted_through"])
        bench = data[data.target_date > cutoff].sort_values(["target_date", "cell_id"]).reset_index(drop=True)
        load_ms = (time.perf_counter() - t0) * 1000

        # Score every replay day with the teammate's own infer(), then check our feature copy matches it.
        t1 = time.perf_counter()
        bench["score"] = infer(self.bundle, bench)
        score_ms = (time.perf_counter() - t1) * 1000
        X, baseline = build_features(self.bundle, bench)
        w = self.bundle["weather_weight"]
        mine = (1 - w) * baseline + w * self.bundle["model"].predict_proba(X)[:, 1]
        assert np.allclose(mine, bench.score.to_numpy(), atol=1e-12, rtol=0), "feature assembly drifted from predict.infer"
        for col in X.columns:
            if col not in bench:
                bench[col] = X[col].to_numpy()
        bench["rank"] = bench.groupby("target_date").score.rank(method="first", ascending=False).astype(int)
        threshold = float(self.bundle["threshold"])
        bench["alert"] = bench.score >= threshold
        self.bench, self.threshold = bench, threshold

        y = bench.fire_next_day.to_numpy()
        s = bench.score.to_numpy()

        # Headline metrics, recomputed live from the model (should match REPORT.md section 11).
        top = bench["rank"] <= REVIEW_K
        tp = int(((s >= threshold) & (y == 1)).sum())
        fp = int(((s >= threshold) & (y == 0)).sum())
        fn = int(((s < threshold) & (y == 1)).sum())
        tn = int(((s < threshold) & (y == 0)).sum())
        prevalence = float(y.mean())
        caught = int((top & (y == 1)).sum())
        precision, recall, _ = precision_recall_curve(y, s)
        idx = np.unique(np.linspace(0, len(precision) - 1, 120).astype(int))
        base_only = bench.historical_prior.to_numpy()
        years = []
        for yr, g in bench.groupby(bench.target_date.dt.year):
            gt = g["rank"] <= REVIEW_K
            years.append({
                "year": int(yr),
                "positive_cell_days": int(g.fire_next_day.sum()),
                "fires": int(g.recorded_fires.sum()),
                "ap": clean(average_precision_score(g.fire_next_day, g.score)),
                "top3_recall": clean((gt & (g.fire_next_day == 1)).sum() / g.fire_next_day.sum()),
            })

        t2 = time.perf_counter()
        perm = permutation_importance(self.bundle["model"], X, y, scoring="average_precision", n_repeats=3, random_state=42, n_jobs=1)
        perm_ms = (time.perf_counter() - t2) * 1000
        imp = {}
        for name, val in zip(X.columns, perm.importances_mean):
            label = FEATURE_LABELS.get(name, name)
            imp[label] = imp.get(label, 0.0) + float(val)
        importance = sorted(({"feature": k, "importance": round(v, 5)} for k, v in imp.items()), key=lambda d: -d["importance"])[:12]

        # Per-day timeline for the season chart and the play-through.
        days = bench.assign(top_hit=top & (bench.fire_next_day == 1)).groupby("target_date").agg(
            fires=("recorded_fires", "sum"), fire_cells=("fire_next_day", "sum"),
            max_score=("score", "max"), mean_score=("score", "mean"), caught=("top_hit", "sum"),
        )

        fires = pd.read_csv(PROJECT / "datasets/included_fires.csv", parse_dates=["target_date"])
        fires = fires[(fires.target_date > cutoff)]
        self.fires = {d.strftime("%Y-%m-%d"): g[["latitude", "longitude", "cell_id"]].round(4).values.tolist()
                      for d, g in fires.groupby("target_date")}

        cells = []
        for cid, code in sorted(self.bundle["history"]["cell_codes"].items(), key=lambda kv: kv[1]):
            r, c = int(cid[1:cid.index("c")]), int(cid[cid.index("c") + 1:])
            south, west = LAT0 + r * STEP, LON0 + c * STEP
            cells.append({"id": cid, "bounds": [[south, west], [south + STEP, west + STEP]],
                          "fires_total": int(bench.loc[bench.cell_id == cid, "recorded_fires"].sum())})

        # Time one live day of inference (24 rows), the way the website calls it.
        sample = bench[bench.target_date == bench.target_date.iloc[-1]]
        reps = []
        for _ in range(20):
            t = time.perf_counter()
            infer(self.bundle, sample)
            reps.append((time.perf_counter() - t) * 1000)

        self.overview = {
            "model": {
                "name": self.bundle["selected_name"], "version": self.bundle["version"],
                "learner": type(self.bundle["model"]).__name__, "trees": int(self.bundle["model"].n_iter_),
                "features": len(self.bundle["features"]), "weather_weight": w, "threshold": threshold,
                "trained_through": str(self.bundle["history"]["fitted_through"])[:10],
                "sha256": manifest.get("sha256"), "size_bytes": (PROJECT / "model/model.joblib").stat().st_size,
            },
            "dates": {"first": bench.target_date.min().strftime("%Y-%m-%d"), "last": bench.target_date.max().strftime("%Y-%m-%d")},
            "cells": cells,
            "data": {
                "fire_records_total": int(sum(1 for _ in open(PROJECT / "datasets/included_fires.csv")) - 1),
                "training_rows": int((data.target_date <= cutoff).sum()),
                "replay_rows": int(len(bench)), "replay_fires": int(bench.recorded_fires.sum()),
                "first_year": int(data.target_date.dt.year.min()), "last_year": int(data.target_date.dt.year.max()),
            },
            "metrics": {
                "rows": int(len(y)), "positives": int(y.sum()), "prevalence": clean(prevalence),
                "average_precision": clean(average_precision_score(y, s)), "roc_auc": clean(roc_auc_score(y, s)),
                "baseline_average_precision": clean(average_precision_score(y, base_only)),
                "brier": clean(np.mean((s - y) ** 2)),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                "threshold_precision": clean(tp / (tp + fp)), "threshold_recall": clean(tp / (tp + fn)),
                "accuracy": clean((tp + tn) / len(y)), "always_no_accuracy": clean(1 - prevalence),
                "review_k": REVIEW_K, "review_caught": caught, "review_recall": clean(caught / y.sum()),
                "review_random": REVIEW_K / len(cells),
                "pr_curve": [[clean(recall[i]), clean(precision[i])] for i in idx],
                "years": years,
            },
            "importance": importance,
            "timeline": {
                "dates": [d.strftime("%Y-%m-%d") for d in days.index],
                "fires": days.fires.astype(int).tolist(), "fire_cells": days.fire_cells.astype(int).tolist(),
                "max_score": [round(float(v), 4) for v in days.max_score],
                "caught": days.caught.astype(int).tolist(),
            },
            "runtime": {
                "machine": platform.machine(), "cpus": os.cpu_count(), "python": platform.python_version(),
                "system": f"{platform.system()} {platform.release()}",
                "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "load_ms": round(load_ms, 1), "score_all_ms": round(score_ms, 1),
                "score_day_ms": round(float(np.median(reps)), 2), "importance_ms": round(perm_ms, 1),
            },
        }
        log(f"ready on {platform.machine()}: {len(bench)} rows scored in {score_ms:.0f} ms, AP={self.overview['metrics']['average_precision']}")

    def day(self, day: str) -> dict:
        rows = self.bench[self.bench.target_date == pd.Timestamp(day)]
        if rows.empty:
            raise KeyError(day)
        t = time.perf_counter()
        live = infer(self.bundle, rows)  # re-score live on LinuxONE so the page shows real inference time
        live_ms = (time.perf_counter() - t) * 1000
        cells = []
        for r, score in zip(rows.itertuples(), live):
            cells.append({
                "id": r.cell_id, "score": clean(score), "rank": int(r.rank), "alert": bool(score >= self.threshold),
                "fires": int(r.recorded_fires), "past_rate": clean(r.historical_prior),
                "weather": {"tmax": clean(r.tmax_lag1), "rh": clean(r.rhmin_lag1), "wind": clean(r.windmax_lag1),
                            "rain30": clean(r.rain_sum30), "dry_days": int(r.dry_days)},
                "anomaly": {"heat": clean(r.tmax_lag1_anomaly), "humidity": clean(r.rhmin_lag1_anomaly),
                            "wind": clean(r.windmax_lag1_anomaly), "rain30": clean(r.rain_sum30_anomaly)},
                "reasons": reasons_for(r),
            })
        cells.sort(key=lambda c: c["rank"])
        top = cells[:REVIEW_K]
        return {
            "date": day, "issued": (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "live_ms": round(live_ms, 2), "machine": platform.machine(),
            "cells": cells, "fires": self.fires.get(day, []),
            "summary": {"fire_cells": sum(c["fires"] > 0 for c in cells), "fires": sum(c["fires"] for c in cells),
                        "caught": sum(c["fires"] > 0 for c in top), "alerts": sum(c["alert"] for c in cells)},
        }


OUTLOOK: Outlook | None = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):  # keep container logs quiet except errors
        if args and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/health":
            return self.send_json({"status": "ok", "machine": platform.machine()})
        if url.path == "/api/overview":
            return self.send_json(OUTLOOK.overview)
        if url.path == "/api/day":
            day = (parse_qs(url.query).get("date") or [""])[0]
            try:
                date.fromisoformat(day)
                return self.send_json(OUTLOOK.day(day))
            except (ValueError, KeyError):
                return self.send_json({"error": "Pick a date between 2022-01-01 and 2024-12-31."}, HTTPStatus.BAD_REQUEST)
        if url.path.startswith("/api/"):
            return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        return super().do_GET()


if __name__ == "__main__":
    OUTLOOK = Outlook()
    log(f"listening on http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
