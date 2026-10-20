# Multi-Agent Precedence Control for Mixed Passenger–Freight Rail Corridors

Six tabular RL agents, one per station on an 870 km Kanpur → Asansol corridor model, decide whether an approaching
freight should **PROCEED**, take the station **LOOP** so a faster passenger train can overtake, or **HOLD** at the signal.
The agents learn independently (IQL and SARSA) from a shared team reward. They are compared against FCFS,
passenger-first and random dispatching on held-out timetables, with bootstrap CIs and paired permutation tests.

## Try your own scenario

The **Simulate** page (`/#/simulate`) runs the simulator live on the server. You set the timetable seed, traffic
load, passenger share and class mix, how many trains arrive late and by how much, departure jitter, which trained
model run the agents use, and the scoring weights. FCFS, passenger-first, IQL and SARSA then dispatch the same
generated day, and the page shows their metrics side by side with a replay of each. Results are deterministic in the
inputs, and the page URL encodes them, so a link reproduces the same run. The same is available as JSON at
`GET /api/simulate?seed=1&load=1.35&passenger_share=0.6&...` (see `api/main.py` for every parameter and its range).

## Quick start

```bash
pip install -r requirements.txt
cd web && npm ci && cd ..

python -m pytest -q                 # 18 tests: simulator invariants + learning sanity
python -m experiments.train         # ~9 min with 11 worker processes: 10 seeds x {main, 2 ablations} x {IQL, SARSA} + sweep
python -m experiments.evaluate      # ~15 s: held-out eval, stats, behaviour analysis, replays -> results/*.json
cd web && npm run build && cd ..
python -m uvicorn api.main:app --port 8000   # open http://127.0.0.1:8000
```

The same steps are available as `make all` / `make serve` where `make` exists. For UI development, run `npm run dev`
in `web/`. It serves on :5173 and proxies `/api` to :8000.

## Deploy (Render)

The deployed service only serves precomputed results. Training and evaluation run locally, and the output in
`results/` is committed to the repo and baked into the image.

1. Push this folder to a GitHub or GitLab repo, including `results/`.
2. In Render, choose **New → Blueprint** and pick the repo. `render.yaml` defines one Docker web service with a health
   check on `/api/health`. Alternatively, choose **New → Web Service**, then **Language: Docker**, and leave the
   defaults.
3. To update after retraining, run `python -m experiments.train` and `python -m experiments.evaluate`, then commit
   `results/` and push. Render redeploys automatically.

The `Dockerfile` builds the UI in a Node stage and runs `uvicorn` on Render's `$PORT` in a slim Python image. The image
is about 340 MB. It contains the simulator and the 20 main-run Q-tables needed for live simulation, but not the
ablation or sweep models. To test locally:

```bash
docker build -t rail-precedence . && docker run -p 8000:8000 rail-precedence
```

On Render's free plan the service sleeps when idle, so the first request after a while takes about a minute.

## Layout

| Path | Contents |
|---|---|
| `config.yaml` | Every parameter and seed. Nothing else holds magic numbers. |
| `sim/` | Corridor geometry, train classes, timetable generator calibrated to a utilisation band, fixed-block environment |
| `agents/` | State discretiser, tabular Q-learning / SARSA, independent-learner wrapper, heuristic baselines |
| `experiments/` | `train.py` (parallel training + learning curves), `evaluate.py` (held-out protocol), `ablations.py`, `stats.py` |
| `api/` | FastAPI: serves `results/*.json`, the built UI, and live scenario simulation (`simulate.py`) |
| `web/` | React + Vite UI with hand-written SVG charts: overview, results, learning curves, ablations, sensitivity, behaviour, replay |
| `results/` | Generated. Q-tables, curves, evaluation JSON, replays, logs |
| `DECISIONS.md` | Modelling choices and why |

## Results (current `results/`, 10 runs × 5 held-out timetables)

| Policy | Passenger delay (min) | Freight delay (min) | Team return |
|---|---:|---:|---:|
| IQL | 3,171 [2,957, 3,413] | 1,509 | −40,829 |
| SARSA | 3,219 [2,940, 3,520] | 1,460 | −41,485 |
| Passenger-first | 3,041 [2,816, 3,289] | 2,002 | −38,931 |
| FCFS | 4,719 [4,545, 4,906] | 4 | −56,952 |
| Random | 4,276 [4,030, 4,516] | 732 | −52,919 |

- **The learned agents do not beat passenger-first.** On team return, IQL is −1,899 behind (95% CI [−3,291, −448]) and
  SARSA −2,555 behind. Neither gap is significant after Holm correction (p = 0.12 and 0.14). On passenger delay they are
  also slightly behind. They do trade about 500 min less freight delay for that.
- Both learned agents beat FCFS by a wide margin, about 16,000 return and 1,500 passenger-min (Holm p = 0.010, the
  smallest p attainable with 10 paired runs). IQL and SARSA are indistinguishable from each other (p = 0.64).
- **Ablation A (local-only reward) hurts.** Team return drops by about 3,500 for both algorithms (p = 0.006).
  Passenger delay rises by about 340 min while freight delay falls by about 370 min, so agents optimising their own
  section push passenger delay downstream.
- **Ablation B (no downstream signal) shows no measurable effect.** The differences are within ±1,100 return and
  p ≥ 0.6. Within this training budget the congestion feature does not pay for the 3× larger state space.
- **Sensitivity.** Across the α × γ grid (3 seeds per cell) IQL ranges from −38.3k to −44.0k. γ = 0.99 is best in every
  row, but the cell CIs overlap heavily. The default cell computed on 3 seeds (−43.5k) and on all 10 (−40.8k) differ by
  more than most cells differ from each other, so seed variance dominates.
- **Behaviour.** About 70% of the learned agents' decisions concern a freight already standing in a loop, and in every
  one of their most-visited states the seeds vote to keep it there. On approach they hold far more often (about 40% of
  approach decisions) than passenger-first, which never holds. About 3% of their decisions are invalid. In the replays,
  61 of 68 invalid decisions are LOOP requests at a station whose loop is already occupied; the interlocking refuses
  them and charges the penalty.

The **Results** page in the UI shows these numbers with CIs, per-run points and every paired test. **Replay** shows a
time–distance diagram of any policy on three held-out days.

## Reproducibility

All randomness derives from `seed` in `config.yaml`. Training timetables use seed `1,000,000·(run+1) + episode`.
Held-out timetables use `900,000,000 + 1000·run + k`, which never overlaps training and differs between runs. Rerunning
`train` + `evaluate` regenerates the same timetables and, with the same NumPy version, the same results.
