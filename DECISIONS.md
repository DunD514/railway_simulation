# Design decisions

Each entry gives the choice, the reason for it, and what it costs. Values live in `config.yaml`.

## Scope and model

**One direction only (UP, Kanpur → Asansol).** The real corridor is double track. The conflict this project studies is
a fast train catching a slow one on the same line, and that conflict is independent per direction. Modelling one line
halves simulation cost without losing the decision. *Cost:* no crossing or opposing-move conflicts. Single-line
sections would need a different action set.

**Fixed-block signalling, 10 km nominal blocks, one loop per station holding one train.** This is the coarsest model in
which overtaking capacity is genuinely scarce. A block holds one train. The rear block is released only after the head
clears an overlap: the 1.2 km freight braking distance for freight, 0.6 km for passenger stock. Loop moves run at turnout
speed for 2 km on either side. *Cost:* no intermediate loops and no automatic-block headways shorter than 10 km.

**Decisions only on conflict.** An agent is asked for a decision only when a freight is within the 12 km approach zone
and a passenger would come within 5 min of it before the next loop, inside a 120 min look-ahead. That window covers the
catch-up horizon on the longest (200 km) section. Asking every epoch would drown the learning signal in trivial
PROCEEDs.

**Interlocking guards.** At most 2 consecutive HOLDs (10 min); after that a further HOLD is ignored. A LOOP into an
occupied or reserved loop, or with no braking room, is refused. The train is held instead and the agent pays a penalty
of 100. The agent still learns the cost of the action, and no action can produce an unsafe state.

## Timetables

**Synthetic, calibrated to 120–150% bottleneck utilisation.** Utilisation is a compression-style headway measure in the
spirit of UIC 406 (see `sim/timetable.py`). Above 100% the timetable cannot run without overtaking, so precedence
decisions matter in every episode. The mix is 60% passenger (20% Vande Bharat, 30% Rajdhani, 50% Express) to 40%
freight. Trains enter over 12 h with up to 4 min jitter, and 30% arrive late, with an exponential delay of mean 12 min.
A fresh timetable is used every training episode so agents cannot memorise one day.

## Learning

**Independent tabular learners with a shared team reward.** Each station has its own Q-table over a small discrete
state of 1,440 states, so policies can be inspected state by state (Behaviour page). The shared reward
`−(10·priority-weighted passenger delay + 1·freight delay)` removes the incentive to dump delay on the next section.
Ablation A tests this directly.

**SMDP discounting, γ per 5-minute epoch.** Decisions are irregular in time, so the bootstrap target is discounted by
the elapsed epochs between an agent's consecutive decisions, with reward accumulated over the gap.

**Neutral, not zero, Q initialisation.** Every reward is negative, so Q₀ = 0 is extremely optimistic. With α = 0.1 it
takes hundreds of visits to wash out, and untried actions keep looking best. Q₀ = r̄/(1−γ), with r̄ ≈ −160 per epoch
(the typical team reward under heuristic play), starts every action near its true scale. For local rewards r̄ is
divided by the number of agents.

**500 training episodes × 10 seeds.** This keeps `make all` within about 30 minutes on a laptop; training currently takes
about 9 min with 11 worker processes. *Cost:* the learning curves are still rising at episode 500 (Learning page). The learned agents'
gap to passenger-first may be partly a budget effect rather than a limit of the method.

**State features.** ETA of the most urgent conflicting passenger (5 bins); freight distance to station, or already in
loop (4); freight stopped or moving (2); braking room for the turnout (2); loop free or busy (2); own-section passenger
delay last epoch (3); downstream congestion message (3). The congestion levels are cut at roughly the terciles seen under baseline play. Ablation B
drops the downstream feature (480 states).

## Evaluation protocol

- **The unit of analysis is the training run.** Each of the 10 runs is evaluated greedily on its own 5 held-out
  timetables, and all policies are evaluated on the same timetables. Run means go into the statistics, so the n is 10,
  not 50.
- **CIs** are percentile bootstrap (10,000 resamples) over run means.
- **Tests** are exact two-sided paired sign-flip permutation tests over runs, Holm-corrected within each metric across
  the five main comparisons. With n = 10 the smallest attainable p is 2/1024, which is 0.0098 after a ×5 Holm factor.
  CIs are not adjusted, so an interval can exclude zero while the adjusted p is above 0.05. The UI notes this.
- **Verdict sentences are generated from the tests** (`experiments/evaluate.py::verdicts`) and never written by hand,
  so the UI and README cannot overstate a result.
- **Sweep cells use 3 seeds.** Their CIs are wide, and the sweep is meant to show sensitivity, not to tune. The default
  cell reuses the first 3 main runs.
