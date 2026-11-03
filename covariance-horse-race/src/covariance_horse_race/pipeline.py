"""Pipeline stages. Each stage reads the previous stage's files, so any stage can
be re-run alone, and every run's outputs sit under data/runs/<config fingerprint>
next to the resolved configuration that produced them."""

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from covariance_horse_race.config import ResearchConfig
from covariance_horse_race.data import store
from covariance_horse_race.data.source import FetchReport, fetch_all
from covariance_horse_race.evaluation.backtest import BacktestOutput, run_universe
from covariance_horse_race.evaluation.simulation import run_simulation
from covariance_horse_race.paths import ProjectPaths
from covariance_horse_race.report import readme
from covariance_horse_race.report.results import build_results, write_results

Log = Callable[[str], None]


def run_dir(paths: ProjectPaths, config: ResearchConfig) -> Path:
    directory = paths.runs / config.fingerprint()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.resolved.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    return directory


def fetch(
    config: ResearchConfig, paths: ProjectPaths, mirror: Path | None, log: Log | None = None
) -> FetchReport:
    paths.ensure()
    return fetch_all(config.data, paths.raw, paths.manifest, mirror=mirror, log=log)


def build(config: ResearchConfig, paths: ProjectPaths) -> None:
    paths.ensure()
    store.build_processed(config, paths.raw, paths.processed, paths.manifest)


def _backtest_one(args: tuple[ResearchConfig, Path, str]) -> BacktestOutput:
    config, root, universe_key = args
    paths = ProjectPaths(root)
    universe = next(u for u in config.universes if u.key == universe_key)
    returns = store.load(paths.processed, universe.file)
    factor_sets = {
        "primary": store.load_factor_set(paths.processed, config.factors.primary),
        "extended": store.load_factor_set(paths.processed, config.factors.extended),
    }
    return run_universe(config, universe.key, returns, factor_sets, progress=print)


def backtest(config: ResearchConfig, paths: ProjectPaths, workers: int | None, log: Log) -> None:
    directory = run_dir(paths, config)
    started = time.perf_counter()
    jobs = [(config, paths.root, u.key) for u in config.universes]
    n_workers = max(1, min(workers or (os.cpu_count() or 1), len(jobs)))
    log(f"backtesting {len(jobs)} universes on {n_workers} worker(s)")
    if n_workers == 1:
        outputs = [_backtest_one(job) for job in jobs]
    else:
        # "spawn" is the only start method on Windows; using it everywhere means
        # Linux CI exercises exactly the path a Windows machine runs.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as pool:
            outputs = list(pool.map(_backtest_one, jobs))
    monthly = pd.concat([o.monthly for o in outputs], ignore_index=True)
    daily = pd.concat([o.daily for o in outputs], ignore_index=True)
    monthly.to_parquet(directory / "monthly.parquet")
    daily.to_parquet(directory / "daily.parquet")
    _record_timing(directory, "backtest", time.perf_counter() - started, n_workers)
    log(f"wrote {len(monthly)} model-months to {directory}")


def _record_timing(directory: Path, stage: str, seconds: float, workers: int) -> None:
    """Measured wall time of a stage, so the README quotes runtimes instead of guessing."""
    path = directory / "timing.json"
    timing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    timing[stage] = {
        "seconds": round(seconds, 1),
        "workers": workers,
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")


def _data_truths(config: ResearchConfig, paths: ProjectPaths) -> dict[int, np.ndarray]:
    """Full-sample covariance of each real universe, keyed by its dimension."""
    truths: dict[int, np.ndarray] = {}
    for universe in config.universes:
        frame = store.load(paths.processed, universe.file).dropna()
        x = frame.to_numpy(dtype=float)
        if x.shape[1] in config.simulation.dimensions:
            xd = x - x.mean(axis=0)
            truths[x.shape[1]] = xd.T @ xd / (x.shape[0] - 1)
    return truths


def simulate(config: ResearchConfig, paths: ProjectPaths, log: Log) -> None:
    directory = run_dir(paths, config)
    truths = _data_truths(config, paths) if config.simulation.truth.source == "data" else None
    started = time.perf_counter()
    frame = run_simulation(config, truths=truths, progress=log)
    frame.to_parquet(directory / "simulation.parquet")
    _record_timing(directory, "simulate", time.perf_counter() - started, 1)


def report(config: ResearchConfig, paths: ProjectPaths, log: Log) -> None:
    directory = paths.runs / config.fingerprint()
    if not (directory / "monthly.parquet").is_file():
        raise FileNotFoundError(
            f"no backtest for configuration {config.fingerprint()}: run `chr backtest` "
            "with the same config and overrides first"
        )
    monthly = pd.read_parquet(directory / "monthly.parquet")
    daily = pd.read_parquet(directory / "daily.parquet")
    sim_path = directory / "simulation.parquet"
    simulation = pd.read_parquet(sim_path) if sim_path.is_file() else pd.DataFrame()
    results = build_results(config, paths, monthly, daily, simulation)
    timing_path = directory / "timing.json"
    if timing_path.is_file():
        results["runtime"] = json.loads(timing_path.read_text(encoding="utf-8"))
    write_results(results, paths.results)
    readme.write(paths.readme_template, results, paths.readme)
    readme.write(paths.paper_template, results, paths.paper)
    log(f"wrote {paths.results.relative_to(paths.root)}, README.md and paper/paper.qmd")
