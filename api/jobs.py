"""Background jobs that compute every page of the app for a user scenario.

Two kinds of job, one at a time, on a shared process pool:
- ``eval``: evaluate a set of trained agents on the scenario's held-out days (Results, Ablations, Sensitivity,
  Behaviour, Replay, Overview). ``agents`` is "default" (agents trained on config.yaml) or "trained" (agents
  trained on this scenario).
- ``train``: train IQL and SARSA on the scenario's own days with a chosen budget (Learning curves and the
  "trained" agents), then evaluate them.

Outputs go to results/scenarios/<key>/{train,eval_default,eval_trained}/ with the same file layout as results/.
The default scenario is config.yaml itself, so its pages are the precomputed files in results/.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from agents.baselines import FCFSPolicy, run_policy_episode
from api.scenario import Scenario, base, describe, scenario_config, traffic_key
from experiments.common import RESULTS, init_progress, n_workers
from experiments.evaluate import available_runs, evaluate_all
from experiments.train import plan_jobs, train_all
from sim.env import CorridorEnv
from sim.timetable import generate_timetable

SCENARIOS = RESULTS / "scenarios"
BUDGET_LIMITS = {"episodes": (20, 500), "seeds": (1, 10)}


@dataclass
class Budget:
    """How much training a ``train`` job does."""

    episodes: int = 100
    seeds: int = 3
    ablations: bool = False
    sweep: bool = False

    def validate(self) -> "Budget":
        for k, (lo, hi) in BUDGET_LIMITS.items():
            if not lo <= getattr(self, k) <= hi:
                raise ValueError(f"{k} must be between {lo} and {hi}")
        return self


@dataclass
class Job:
    id: str
    kind: str                      # "eval" | "train"
    key: str                       # scenario traffic key
    agents: str                    # "default" | "trained" (eval jobs)
    scenario: Scenario
    budget: Budget | None = None
    state: str = "queued"          # queued | running | done | failed
    done: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    _parts: dict[str, int] = field(default_factory=dict, repr=False)

    def public(self) -> dict[str, Any]:
        elapsed = (self.finished or time.time()) - self.started if self.started else 0.0
        eta = None
        if self.state == "running" and self.done > 0 and self.total > 0:
            eta = elapsed * (self.total - self.done) / self.done
        return {
            "id": self.id, "kind": self.kind, "key": self.key, "agents": self.agents, "state": self.state,
            "done": self.done, "total": self.total, "fraction": (self.done / self.total) if self.total else 0.0,
            "elapsed_s": round(elapsed, 1), "eta_s": round(eta, 1) if eta is not None else None,
            "message": self.message, "error": self.error,
            "budget": asdict(self.budget) if self.budget else None,
        }


# ---------------------------------------------------------------- where things live

def train_dir(key: str) -> Path:
    return RESULTS if key == "default" else SCENARIOS / key / "train"


def eval_dir(key: str, agents: str) -> Path:
    return RESULTS if key == "default" else SCENARIOS / key / f"eval_{agents}"


def is_trained(key: str) -> bool:
    d = train_dir(key)
    return (d / "curves_summary.json").is_file() and available_runs(d / "models", "main") > 0 and \
        (key == "default" or (d / "done.json").is_file())


def train_info(key: str) -> dict[str, Any] | None:
    """Budget the scenario's agents were trained with (None if untrained)."""
    if not is_trained(key):
        return None
    if key == "default":
        cfg, _ = base()
        e = cfg["experiment"]
        return {"episodes": int(e["train_episodes"]), "seeds": int(e["n_seeds"]), "ablations": True, "sweep": True,
                "precomputed": True}
    return json.loads((train_dir(key) / "done.json").read_text(encoding="utf-8"))


def is_evaluated(key: str, agents: str) -> bool:
    if key == "default":
        return (RESULTS / "eval_main.json").is_file()
    return (eval_dir(key, agents) / "done.json").is_file()


def models_for(key: str, agents: str) -> tuple[Path, Path]:
    """(models dir, curves_summary path) of the agents used for a scenario."""
    d = RESULTS if agents == "default" else train_dir(key)
    return d / "models", d / "curves_summary.json"


# ---------------------------------------------------------------- manager

class JobManager:
    """Single worker thread running jobs in FIFO order on one long-lived process pool."""

    def __init__(self) -> None:
        cfg, _ = base()
        env = os.environ.get("RAIL_WORKERS")
        self.workers = max(1, int(env)) if env else n_workers(cfg)
        self.jobs: dict[str, Job] = {}
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()
        self.current: Job | None = None
        self._pool = None
        self._progress_q = None
        self._sec_per_episode: float | None = None
        threading.Thread(target=self._run, name="jobs", daemon=True).start()

    # -- pool
    def _get_pool(self):
        if self._pool is None:
            ctx = get_context("spawn")
            self._progress_q = ctx.Queue()
            self._pool = ctx.Pool(self.workers, initializer=init_progress, initargs=(self._progress_q,))
            threading.Thread(target=self._drain, name="progress", daemon=True).start()
        return self._pool

    def _drain(self) -> None:
        while True:
            try:
                key, done, _total = self._progress_q.get()
            except (EOFError, OSError):
                return
            job = self.current
            if job is not None and job.kind == "train":
                job._parts[key] = done
                job.done = sum(job._parts.values())

    # -- public API
    def submit(self, kind: str, sc: Scenario, agents: str = "default", budget: Budget | None = None) -> Job:
        key = traffic_key(sc)
        jid = f"{kind}:{key}:{agents}" if kind == "eval" else f"train:{key}"
        with self.lock:
            job = self.jobs.get(jid)
            if job and job.state in ("queued", "running"):
                return job
            if job and job.state == "failed" and time.time() - job.finished < 60:
                return job  # report the failure instead of retrying in a loop; retry allowed after a minute
            # a train job also invalidates and re-queues evaluation of the trained agents
            job = Job(jid, kind, key, agents, sc, budget)
            self.jobs[jid] = job
            self.queue.put(jid)
            return job

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def active_for(self, key: str) -> list[Job]:
        return [j for j in self.jobs.values() if j.key == key and j.state in ("queued", "running")]

    def queue_position(self, job: Job) -> int:
        ahead = [j for j in self.jobs.values() if j.state == "queued" and j.created < job.created]
        return len(ahead) + (1 if self.current and self.current is not job else 0)

    def estimate_train_seconds(self, sc: Scenario, budget: Budget) -> dict[str, Any]:
        cfg = self._budget_cfg(sc, budget)
        jobs, _ = plan_jobs(cfg, Path("."), budget.ablations, budget.sweep)
        episodes = len(jobs) * budget.episodes
        spe = self._episode_seconds()
        return {"episodes": episodes, "jobs": len(jobs), "workers": self.workers,
                # learning episodes cost ~2.7x a heuristic one (measured); +2 s pool start and assembly
                "seconds": round(2.0 + episodes * spe * 2.7 / min(self.workers, len(jobs)), 1)}

    # -- internals
    def _episode_seconds(self) -> float:
        if self._sec_per_episode is None:
            cfg, ctx = base()
            tt = generate_timetable(12345, cfg, ctx.corridor, ctx.classes)
            t0 = time.time()
            for _ in range(3):
                run_policy_episode(CorridorEnv(cfg, tt, ctx.corridor), FCFSPolicy())
            self._sec_per_episode = (time.time() - t0) / 3
        return self._sec_per_episode

    @staticmethod
    def _budget_cfg(sc: Scenario, budget: Budget) -> dict[str, Any]:
        cfg = scenario_config(sc)
        cfg["experiment"]["train_episodes"] = budget.episodes
        cfg["experiment"]["n_seeds"] = budget.seeds
        cfg["experiment"]["sweep"]["n_seeds"] = min(budget.seeds, int(cfg["experiment"]["sweep"]["n_seeds"]))
        return cfg

    def _run(self) -> None:
        while True:
            jid = self.queue.get()
            job = self.jobs[jid]
            if job.state != "queued":
                continue
            self.current = job
            job.state, job.started = "running", time.time()
            try:
                if job.kind == "train":
                    self._train(job)
                else:
                    self._eval(job)
                job.state = "done"
            except Exception as e:  # noqa: BLE001 - reported to the client
                job.state, job.error = "failed", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                job.finished = time.time()
                self.current = None

    def _eval(self, job: Job) -> None:
        out = eval_dir(job.key, job.agents)
        models, curves = models_for(job.key, job.agents)
        if available_runs(models, "main") == 0:
            raise RuntimeError("no trained agents for this scenario yet")
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "scenario.json").write_text(json.dumps(describe(job.scenario)), encoding="utf-8")

        def progress(i: int, n: int) -> None:
            job.done, job.total = i, n

        job.message = "evaluating on held-out days"
        evaluate_all(scenario_config(job.scenario), tmp, models, self._get_pool(), curves_path=curves,
                     log=lambda m: None, progress=progress)
        (tmp / "done.json").write_text(json.dumps({"finished": time.time()}), encoding="utf-8")
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)

    def _train(self, job: Job) -> None:
        assert job.budget is not None and job.key != "default"
        cfg = self._budget_cfg(job.scenario, job.budget)
        out = train_dir(job.key)
        tmp = out.with_name("train.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        jobs, _ = plan_jobs(cfg, tmp / "models", job.budget.ablations, job.budget.sweep)
        job.total, job.done, job._parts = len(jobs) * job.budget.episodes, 0, {}
        job.message = "training"
        train_all(cfg, tmp, self._get_pool(), ablations=job.budget.ablations, sweep=job.budget.sweep,
                  log=lambda m: None)
        info = {**asdict(job.budget), "finished": time.time()}
        (tmp / "done.json").write_text(json.dumps(info), encoding="utf-8")
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
        # the trained agents changed: evaluate them on this scenario
        shutil.rmtree(eval_dir(job.key, "trained"), ignore_errors=True)
        job.done = job.total
        self.submit("eval", job.scenario, "trained")


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
