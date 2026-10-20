"""FastAPI service: every page of the app for any user scenario, plus the built web UI.

Run: uvicorn api.main:app --port 8000

Scenario inputs are query parameters (see api/scenario.py LIMITS). The default scenario is config.yaml, whose pages
are the precomputed files in results/. Any other scenario is computed on demand by background jobs
(api/jobs.py): a page that is not ready yet answers 202 with the job to poll.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.jobs import (
    BUDGET_LIMITS, Budget, eval_dir, is_evaluated, is_trained, manager, models_for, train_dir, train_info,
)
from api.scenario import LIMITS, Scenario, ScenarioError, describe, make_scenario, scenario_config, traffic_key
from api.simulate import simulate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Rail precedence control", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)  # result JSON is up to ~1 MB, compresses ~5x

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")
EVAL_PAGES = {
    "eval": "eval_main.json", "ablations": "ablations.json", "sensitivity": "sensitivity.json",
    "behavior": "behavior.json", "behaviour": "behavior.json", "replays": "replays/index.json",
}


def _read(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def scenario_dep(
    seed: int = Query(1, ge=0, le=2**31 - 1),
    load: float = Query(LIMITS["load"][0], ge=LIMITS["load"][1], le=LIMITS["load"][2]),
    passenger_share: float = Query(LIMITS["passenger_share"][0], ge=LIMITS["passenger_share"][1], le=LIMITS["passenger_share"][2]),
    mix_vb: float = Query(LIMITS["mix_vb"][0], ge=0, le=1),
    mix_raj: float = Query(LIMITS["mix_raj"][0], ge=0, le=1),
    mix_exp: float = Query(LIMITS["mix_exp"][0], ge=0, le=1),
    late_prob: float = Query(LIMITS["late_prob"][0], ge=0, le=1),
    late_mean_min: float = Query(LIMITS["late_mean_min"][0], ge=0, le=LIMITS["late_mean_min"][2]),
    jitter_min: float = Query(LIMITS["jitter_min"][0], ge=0, le=LIMITS["jitter_min"][2]),
    model_run: int = Query(0, ge=0, le=9),
    w_p: float = Query(LIMITS["w_p"][0], ge=0, le=LIMITS["w_p"][2]),
    w_f: float = Query(LIMITS["w_f"][0], ge=0, le=LIMITS["w_f"][2]),
) -> Scenario:
    try:
        return make_scenario(seed=seed, load=load, passenger_share=passenger_share, mix_vb=mix_vb, mix_raj=mix_raj,
                             mix_exp=mix_exp, late_prob=late_prob, late_mean_min=late_mean_min, jitter_min=jitter_min,
                             model_run=model_run, w_p=w_p, w_f=w_f)
    except ScenarioError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


AgentsQ = Query("default", pattern="^(default|trained)$")


def _pending(job: Any) -> JSONResponse:
    m = manager()
    return JSONResponse(status_code=202, content={"status": "computing", "job": job.public(),
                                                   "queue_position": m.queue_position(job)})


def _needs_training(sc: Scenario) -> JSONResponse:
    # 200 with a status field rather than 409: this is an expected state, not a failed request
    return JSONResponse(status_code=200, content={
        "status": "needs_training",
        "detail": "no agents have been trained on this scenario yet - start training or use the default agents",
        "scenario": describe(sc)})


def _eval_file(sc: Scenario, agents: str, rel: str) -> Any:
    """Serve one evaluation output for (scenario, agents), starting the evaluation job if needed."""
    key = traffic_key(sc)
    if key == "default":
        agents = "default"  # the default agents are the ones trained on this scenario
    if agents == "trained" and not is_trained(key):
        return _needs_training(sc)
    if not is_evaluated(key, agents):
        job = manager().submit("eval", sc, agents)
        if job.state == "failed":
            raise HTTPException(status_code=500, detail=f"computing this scenario failed: {job.error}")
        return _pending(job)
    path = eval_dir(key, agents) / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{rel} not found")
    return _read(path)


# ---------------------------------------------------------------- scenario state

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "precomputed": (RESULTS / "eval_main.json").is_file(), "workers": manager().workers}


@app.get("/api/scenario")
def scenario_status(sc: Scenario = Depends(scenario_dep)) -> dict[str, Any]:
    """What exists for a scenario: evaluations, trained agents, running jobs."""
    key = traffic_key(sc)
    m = manager()
    return {
        "scenario": describe(sc),
        "trained": train_info(key),
        "evaluated": {"default": is_evaluated(key, "default"),
                      "trained": is_evaluated(key, "trained") if key != "default" else True},
        "jobs": [j.public() for j in m.active_for(key)],
        "workers": m.workers,
        "budget_limits": BUDGET_LIMITS,
    }


@app.get("/api/jobs/{job_id:path}")
def job_status(job_id: str) -> dict[str, Any]:
    job = manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {**job.public(), "queue_position": manager().queue_position(job)}


def _budget(episodes: int, seeds: int, ablations: bool, sweep: bool) -> Budget:
    try:
        return Budget(episodes, seeds, ablations, sweep).validate()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/api/train/estimate")
def train_estimate(
    sc: Scenario = Depends(scenario_dep),
    episodes: int = Query(100), seeds: int = Query(3), ablations: bool = Query(False), sweep: bool = Query(False),
) -> dict[str, Any]:
    return manager().estimate_train_seconds(sc, _budget(episodes, seeds, ablations, sweep))


@app.post("/api/train")
def train(
    sc: Scenario = Depends(scenario_dep),
    episodes: int = Query(100), seeds: int = Query(3), ablations: bool = Query(False), sweep: bool = Query(False),
) -> JSONResponse:
    """Train IQL and SARSA on this scenario's own days, then evaluate them."""
    if traffic_key(sc) == "default":
        raise HTTPException(status_code=409, detail="the default scenario is already trained (precomputed results)")
    job = manager().submit("train", sc, budget=_budget(episodes, seeds, ablations, sweep))
    return _pending(job)


# ---------------------------------------------------------------- pages

@lru_cache(maxsize=32)
def _overview(sc: Scenario) -> dict[str, Any]:
    from experiments.common import Context
    from experiments.evaluate import overview

    cfg = scenario_config(sc)
    return overview(cfg, Context(cfg))


@app.get("/api/page/overview")
def page_overview(sc: Scenario = Depends(scenario_dep)) -> Any:
    """Corridor, traffic and timetable statistics of the scenario's held-out days (computed immediately)."""
    if traffic_key(sc) == "default" and (RESULTS / "overview.json").is_file():
        return _read(RESULTS / "overview.json")
    try:
        return _overview(Scenario(**{**describe(sc)["all"], "seed": 1, "model_run": 0}))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"could not build timetables for this scenario ({e})") from e


@app.get("/api/page/curves")
def page_curves(sc: Scenario = Depends(scenario_dep), agents: str = AgentsQ) -> Any:
    """Learning curves of the agents in use (default agents: training on config.yaml traffic)."""
    key = traffic_key(sc)
    if agents == "default" or key == "default":
        return _read(RESULTS / "curves_summary.json")
    if not is_trained(key):
        job = manager().get(f"train:{key}")
        if job and job.state in ("queued", "running"):
            return _pending(job)
        return _needs_training(sc)
    return _read(train_dir(key) / "curves_summary.json")


@app.get("/api/page/{name}")
def page(name: str, sc: Scenario = Depends(scenario_dep), agents: str = AgentsQ) -> Any:
    if name not in EVAL_PAGES:
        raise HTTPException(status_code=404, detail="unknown page")
    return _eval_file(sc, agents, EVAL_PAGES[name])


@app.get("/api/page/replay/{policy}/{day}")
def page_replay(policy: str, day: int, sc: Scenario = Depends(scenario_dep), agents: str = AgentsQ) -> Any:
    """One recorded held-out day (``day`` is its timetable seed, as listed by /api/page/replays)."""
    if not _SAFE.match(policy):
        raise HTTPException(status_code=400, detail="bad policy")
    return _eval_file(sc, agents, f"replays/{policy}_{day}.json")


@app.get("/api/simulate")
def simulate_scenario(sc: Scenario = Depends(scenario_dep), agents: str = AgentsQ) -> Any:
    """Simulate one day of the scenario live with every policy. Deterministic in its inputs."""
    key = traffic_key(sc)
    if agents == "trained" and key != "default" and not is_trained(key):
        return _needs_training(sc)
    models, _ = models_for(key, "default" if key == "default" else agents)
    try:
        return simulate(sc, str(models), agents)
    except ScenarioError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


# ---------------------------------------------------------------- precomputed (default scenario) endpoints

def _load(rel: str) -> Any:
    path = RESULTS / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{rel} not found - run `make train eval` first")
    return _read(path)


@app.get("/api/overview")
def overview_default() -> Any:
    return _load("overview.json")


@app.get("/api/eval")
def evaluation_default() -> Any:
    return _load("eval_main.json")


@app.get("/api/curves")
def curves_default() -> Any:
    return _load("curves_summary.json")


@app.get("/api/replays")
def replay_index_default() -> Any:
    return _load("replays/index.json")


if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        """Serve the single-page app (client-side routing falls back to index.html)."""
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="unknown API route")
        candidate = (WEB_DIST / path).resolve()
        if path and candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
