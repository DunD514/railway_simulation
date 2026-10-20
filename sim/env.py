"""Multi-agent corridor environment.

One episode simulates the UP line of the corridor in 5-minute decision epochs,
each split into 1-minute movement substeps. Six section-controller agents (one
per station) are queried only when a freight in their approach zone is in
conflict with a faster passenger train; the env then applies PROCEED / LOOP /
HOLD for that freight.

Signalling is fixed-block: a block holds at most one train, a train may enter
the next block only when it is free, and the block in rear is released only once
the train's head is ``overlap_km`` past its exit (1.2 km freight braking distance,
0.6 km for passenger stock).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim.corridor import Corridor, build_corridor
from sim.timetable import Timetable
from sim.trains import DWELL, EXITED, IN_LOOP, RUNNING, WAITING, TrainState

PROCEED, LOOP, HOLD = 0, 1, 2
ACTION_NAMES = ("PROCEED", "LOOP", "HOLD")
N_ACTIONS = 3

_EPS = 1e-9


@dataclass(frozen=True)
class Observation:
    """Local observation of one agent at a decision point (raw, undiscretised)."""

    agent: int
    freight_tid: int
    passenger_tid: int
    eta_min: float          # most urgent conflicting passenger's unobstructed ETA at this station
    dist_km: float          # freight head distance to the station (loop turnout)
    speed_kmh: float        # freight speed during the last substep
    margin_km: float        # braking margin: dist - braking distance at current speed
    loop_busy: int          # 1 if the loop is occupied or already reserved
    local_delay_min: float  # passenger delay-minutes accrued in this agent's section last epoch
    downstream: int         # congestion level (0/1/2) message received from agent i+1
    n_conflicts: int        # number of conflicting passengers inside the window
    in_loop: int            # 1 if the freight is already standing in the loop (decision = depart or stay)


@dataclass
class StepResult:
    """Outcome of one epoch."""

    shared_reward: float
    local_rewards: list[float]
    invalid_agents: list[int]
    passenger_delay: float   # unweighted passenger delay-minutes accrued this epoch
    freight_delay: float
    done: bool


@dataclass
class EpisodeLog:
    """Decision log and replay frames (only filled when recording)."""

    decisions: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    events: list[list[Any]] = field(default_factory=list)


class CorridorEnv:
    """Discrete-time fixed-block corridor simulator with per-station precedence agents."""

    def __init__(
        self,
        cfg: dict[str, Any],
        timetable: Timetable,
        corridor: Corridor | None = None,
        record: bool = False,
        log_decisions: bool = False,
    ) -> None:
        self.cfg = cfg
        self.corridor = corridor or build_corridor(cfg)
        self.timetable = timetable
        self.record = record
        self.log_decisions = log_decisions or record
        t = cfg["time"]
        self.epoch_min = float(t["epoch_min"])
        self.n_sub = int(t["substeps_per_epoch"])
        self.dt = self.epoch_min / self.n_sub
        self.hard_cap = int(t["hard_cap_epochs"])
        ctl = cfg["control"]
        self.window = float(ctl["window_min"])
        self.margin = float(ctl["catchup_margin_min"])
        self.invalid_penalty = float(ctl["invalid_penalty"])
        self.max_holds = int(ctl["max_consecutive_holds"])
        rw = cfg["reward"]
        self.w_p = float(rw["w_p"])
        self.w_f = float(rw["w_f"])
        self.brake_km = float(cfg["trains"]["freight_braking_km"])
        self.v_turn = self.corridor.turnout_speed_kmh / 60.0
        self.cong_edges = [float(x) for x in cfg["discretization"]["congestion_edges"]]
        self._boundary_station = {s.boundary: s.index for s in self.corridor.stations}
        self._section_blocks = [self.corridor.section_blocks(s) for s in range(self.corridor.n_stations + 1)]
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Reset the episode to t=0 with an empty corridor."""
        n_st = self.corridor.n_stations
        self.now = 0.0
        self.epoch = 0
        self.trains = [TrainState(s, n_st) for s in self.timetable.trains]
        self.waiting = sorted(self.trains, key=lambda tr: (tr.sched.ready_min, tr.tid))
        self.active: list[TrainState] = []
        self.occ = [-1] * self.corridor.n_blocks
        self.loop_occ: list[list[int]] = [[] for _ in range(n_st)]
        self.loop_reserved: list[list[int]] = [[] for _ in range(n_st)]
        self.section_pdelay_last = [0.0] * (n_st + 1)
        self.loop_uses = [0] * n_st
        self.decision_count = [0] * n_st
        self.invalid_count = [0] * n_st
        self.action_count = [[0] * N_ACTIONS for _ in range(n_st)]
        self.total_shared_return = 0.0
        self.done = False
        self.log = EpisodeLog()
        self._pending: dict[int, Observation] = {}
        self._pending_epoch = -1
        self._loop_conflict_cache: dict[tuple[int, int], bool] = {}
        if self.record:
            self._record_frame()

    # ------------------------------------------------------------ geometry
    def loop_busy(self, st: int) -> bool:
        """True when the loop at station ``st`` is occupied or reserved to capacity."""
        return len(self.loop_occ[st]) + len(self.loop_reserved[st]) >= self.corridor.loop_capacity

    def _time_to(self, tr: TrainState, km: float) -> float:
        """Unobstructed minutes until ``tr``'s head reaches ``km`` (passenger dwells included)."""
        v = tr.cls.speed_kpm
        if tr.status == WAITING:
            wait = max(0.0, tr.sched.ready_min - self.now)
            pos = 0.0
        else:
            wait = tr.dwell_left if tr.status == DWELL else 0.0
            pos = tr.pos
        t = wait + max(0.0, km - pos) / v
        if tr.cls.is_passenger and tr.cls.dwell_min > 0:
            for st in self.corridor.stations:
                if pos < st.km < km - _EPS and st.index >= tr.next_stop:
                    t += tr.cls.dwell_min
        return t

    def conflicts(self, f: TrainState, st: int) -> list[tuple[float, TrainState]]:
        """Faster passengers behind freight ``f`` that would catch it before the loop after station ``st``.

        Returns (eta at station st, passenger) pairs within the controller window.
        """
        skm = self.corridor.stations[st].km
        next_km = self.corridor.next_loop_km(st)
        vf = f.cls.speed_kpm
        f_pos = f.pos
        eta_f_next = (next_km - f_pos) / vf
        in_loop = f.status == IN_LOOP
        out: list[tuple[float, TrainState]] = []
        for p in self._passengers_upstream():
            if p.cls.speed_kpm <= vf:
                continue
            if p.status == WAITING:
                if p.sched.ready_min - self.now > self.window:
                    continue
            elif in_loop:
                # behind the looped freight = has not yet departed station st
                if p.pos > skm + _EPS or (p.next_stop > st and p.departures[st] >= 0):
                    continue
            elif p.pos >= f_pos - _EPS:
                continue
            eta_st = self._time_to(p, skm)
            if eta_st > self.window:
                continue
            if self._time_to(p, next_km) < eta_f_next + self.margin:
                out.append((eta_st, p))
        out.sort(key=lambda x: (x[0], x[1].tid))
        return out

    def _passengers_upstream(self) -> list[TrainState]:
        out = [p for p in self.active if p.cls.is_passenger and p.status != EXITED]
        for p in self.waiting:
            if p.sched.ready_min - self.now > self.window:
                break
            if p.cls.is_passenger:
                out.append(p)
        return out

    def congestion_message(self, section: int) -> int:
        """Congestion level an agent broadcasts upstream about its own section (0/1/2).

        Score = fraction of the section's blocks occupied + 0.5 if its loop is busy.
        """
        blocks = self._section_blocks[section]
        occ = sum(1 for b in blocks if self.occ[b] != -1) / max(1, len(blocks))
        if section < self.corridor.n_stations and self.loop_busy(section):
            occ += 0.5
        level = 0
        for e in self.cong_edges:
            if occ >= e:
                level += 1
        return level

    # ------------------------------------------------------------ decisions
    def decision_requests(self) -> list[Observation]:
        """Observations for every agent that faces a freight/passenger conflict this epoch."""
        if self._pending_epoch == self.epoch:
            return list(self._pending.values())
        cand: dict[int, TrainState] = {}
        for f in self.active:
            if f.cls.is_passenger or f.status != RUNNING or f.route_loop >= 0:
                continue
            st = f.next_stop
            if st >= self.corridor.n_stations:
                continue
            dist = self.corridor.stations[st].km - f.pos
            if dist > self.corridor.approach_zone_km + _EPS:
                continue
            prev = cand.get(st)
            if prev is None or f.pos > prev.pos:
                cand[st] = f
        pending: dict[int, Observation] = {}
        for st in range(self.corridor.n_stations):
            conf: list[tuple[float, TrainState]] = []
            f = cand.get(st)
            if f is not None:
                conf = self.conflicts(f, st)
            if not conf:
                # no approaching conflict: a freight standing in the loop may need a depart/stay decision
                f = None
                for tid in self.loop_occ[st]:
                    lf = self.trains[tid]
                    if lf.status == IN_LOOP and lf.loop_clear_left <= 0:
                        conf = self.conflicts(lf, st)
                        if conf:
                            f = lf
                            break
            if f is None or not conf:
                continue
            eta, p = conf[0]
            dist = self.corridor.stations[st].km - f.pos
            speed = f.speed_now
            margin = dist - self.brake_km * (speed / f.cls.speed_kmh) ** 2
            pending[st] = Observation(
                agent=st,
                freight_tid=f.tid,
                passenger_tid=p.tid,
                eta_min=eta,
                dist_km=dist,
                speed_kmh=speed,
                margin_km=margin,
                loop_busy=int(self.loop_busy(st)),
                local_delay_min=self.section_pdelay_last[st],
                downstream=self.congestion_message(st + 1),
                n_conflicts=len(conf),
                in_loop=int(f.status == IN_LOOP),
            )
        self._pending = pending
        self._pending_epoch = self.epoch
        return list(pending.values())

    def loop_valid(self, obs: Observation) -> bool:
        """Whether LOOP is admissible: free loop and enough braking room (always valid = stay, if already looped)."""
        if obs.in_loop:
            return True
        return (not self.loop_busy(obs.agent)) and obs.margin_km >= -_EPS

    # ------------------------------------------------------------ stepping
    def step(self, actions: dict[int, int]) -> StepResult:
        """Apply agents' actions for this epoch and simulate ``substeps_per_epoch`` minutes."""
        if self.done:
            raise RuntimeError("episode finished")
        pending = {o.agent: o for o in self.decision_requests()}
        n_st = self.corridor.n_stations
        held_before = {tr.tid for tr in self.active if tr.hold}
        for tr in self.active:
            tr.hold = False
            tr.release = False
        penalties = [0.0] * n_st
        invalid: list[int] = []
        for agent, obs in pending.items():
            a = int(actions.get(agent, PROCEED))
            f = self.trains[obs.freight_tid]
            self.decision_count[agent] += 1
            self.action_count[agent][a] += 1
            valid = True
            timed_out = False
            if obs.in_loop:
                # PROCEED = depart the loop as soon as the main line ahead is free; LOOP/HOLD = stay
                f.release = a == PROCEED
            elif a == LOOP:
                if self.loop_valid(obs):
                    f.route_loop = agent
                    self.loop_reserved[agent].append(f.tid)
                else:
                    valid = False
                    f.hold = True  # loop route refused: train is held at the home signal
                    penalties[agent] += self.invalid_penalty
                    invalid.append(agent)
                    self.invalid_count[agent] += 1
            elif a == HOLD:
                f.hold = True
            if f.hold:
                # interlocking timeout: a freight cannot be held at a signal indefinitely
                if f.tid in held_before and f.hold_streak >= self.max_holds:
                    f.hold = False
                    timed_out = True
                else:
                    f.hold_streak = f.hold_streak + 1 if f.tid in held_before else 1
            if self.log_decisions:
                self.log.decisions.append(
                    {
                        "epoch": self.epoch,
                        "t": round(self.now, 2),
                        "agent": agent,
                        "freight": f.tid,
                        "passenger": obs.passenger_tid,
                        "action": a,
                        "valid": valid,
                        "hold_timed_out": timed_out,
                        "eta": round(obs.eta_min, 2),
                        "dist": round(obs.dist_km, 2),
                        "loop_busy": obs.loop_busy,
                        "downstream": obs.downstream,
                        "in_loop": obs.in_loop,
                    }
                )
        for tr in self.trains:
            tr.accrued_epoch = 0.0
        for _ in range(self.n_sub):
            self._substep()
        # rewards
        n_sec = n_st + 1
        sec_p = [0.0] * n_sec
        sec_pw = [0.0] * n_sec
        sec_f = [0.0] * n_sec
        tot_p = tot_f = tot_pw = 0.0
        for tr in self.trains:
            d = tr.accrued_epoch
            if d <= 0.0:
                continue
            s = self._section_of(tr)
            if tr.cls.is_passenger:
                sec_p[s] += d
                sec_pw[s] += d * tr.cls.priority
                tot_p += d
                tot_pw += d * tr.cls.priority
            else:
                sec_f[s] += d
                tot_f += d
        shared = -(self.w_p * tot_pw + self.w_f * tot_f) - sum(penalties)
        local = [-(self.w_p * sec_pw[i] + self.w_f * sec_f[i]) - penalties[i] for i in range(n_st)]
        self.section_pdelay_last = sec_p
        self.total_shared_return += shared
        self.epoch += 1
        if self.record:
            self._record_frame()
        if self.epoch >= self.hard_cap or all(tr.status == EXITED for tr in self.trains):
            self.done = True
        return StepResult(shared, local, invalid, tot_p, tot_f, self.done)

    def _section_of(self, tr: TrainState) -> int:
        if tr.block < 0:
            return 0
        return self.corridor.block_section[tr.block]

    def _substep(self) -> None:
        dt = self.dt
        t0 = self.now
        # entry: one train per substep into block 0 (FIFO by ready time), at the start of the substep
        if self.waiting and self.waiting[0].sched.ready_min <= t0 + _EPS and self.occ[0] == -1:
            tr = self.waiting.pop(0)
            tr.status = RUNNING
            tr.block = 0
            tr.pos = 0.0
            tr.entered_min = t0
            self.occ[0] = tr.tid
            self.active.append(tr)
            if self.record:
                self.log.events.append([tr.tid, round(t0, 3), 0.0, "enter"])
        # trains that were ready at the start of the substep but could not enter accrue delay
        for tr in self.waiting:
            if tr.sched.ready_min > t0 + _EPS:
                break
            tr.accrued += dt
            tr.accrued_epoch += dt
            tr.accrued_by_section[0] += dt
        looped = [tr for tr in self.active if tr.status == IN_LOOP]
        main = [tr for tr in self.active if tr.status in (RUNNING, DWELL)]
        main.sort(key=lambda tr: (-tr.pos, tr.tid))
        for tr in main:
            self._advance(tr, dt, t0)
        # trains already standing in loops (a train entering a loop this substep starts ticking next substep)
        for tr in looped:
            self._loop_tick(tr, dt, t0)
        self.active = [tr for tr in self.active if tr.status != EXITED]
        self.now = t0 + dt

    def _add_delay(self, tr: TrainState, minutes: float) -> None:
        if minutes > 1e-7:
            tr.accrued += minutes
            tr.accrued_epoch += minutes
            tr.accrued_by_section[self._section_of(tr)] += minutes

    def _advance(self, tr: TrainState, dt: float, t0: float) -> None:
        """Move one main-line train for ``dt`` minutes, honouring signals, dwells and turnouts."""
        c = self.corridor
        bounds = c.bounds
        vline = tr.cls.speed_kpm
        if tr.status == RUNNING and tr.route_loop < 0 and tr.from_loop_until_km <= tr.pos:
            # fast path: cruising inside a block at line speed, no boundary reached this substep
            new_pos = tr.pos + vline * dt
            if new_pos < bounds[tr.block + 1] - _EPS:
                tr.pos = new_pos
                tr.stopped = False
                tr.speed_now = tr.cls.speed_kmh
                if tr.tail_block >= 0 or tr.loop_station >= 0:
                    self._release_rear(tr)
                return
        remaining = dt
        sched_progress = 0.0
        start_pos = tr.pos
        guard = 0
        while remaining > _EPS and guard < 50:
            guard += 1
            if tr.status == DWELL:
                use = min(remaining, tr.dwell_left)
                tr.dwell_left -= use
                remaining -= use
                sched_progress += use
                if tr.dwell_left <= _EPS:
                    tr.dwell_left = 0.0
                    tr.status = RUNNING
                continue
            b = tr.block
            end = bounds[b + 1]
            v = vline
            limit = end
            if tr.route_loop >= 0:
                zone_start = c.stations[tr.route_loop].km - c.turnout_zone_km
                if tr.pos >= zone_start - _EPS:
                    v = min(v, self.v_turn)
                elif zone_start < limit:
                    limit = zone_start
            if tr.from_loop_until_km > tr.pos + _EPS:
                v = min(v, self.v_turn)
                limit = min(limit, tr.from_loop_until_km)
            dist = limit - tr.pos
            if dist > _EPS:
                t_need = dist / v
                if t_need >= remaining:
                    tr.pos += v * remaining
                    sched_progress += v * remaining / vline
                    remaining = 0.0
                else:
                    tr.pos = limit
                    sched_progress += dist / vline
                    remaining -= t_need
                self._release_rear(tr)
            if tr.pos >= end - _EPS:
                tr.pos = end
                if not self._cross_boundary(tr, t0 + dt - remaining):
                    break
        moved = tr.pos - start_pos
        tr.stopped = moved <= _EPS and tr.status != DWELL
        tr.speed_now = 0.0 if tr.stopped else (moved / dt) * 60.0
        if tr.status == EXITED:
            return
        self._add_delay(tr, max(0.0, dt - remaining - sched_progress) + remaining)

    def _release_rear(self, tr: TrainState) -> None:
        bounds = self.corridor.bounds
        if tr.tail_block >= 0 and tr.pos >= bounds[tr.tail_block + 1] + tr.cls.overlap_km - _EPS:
            self.occ[tr.tail_block] = -1
            tr.tail_block = -1
        if tr.loop_station >= 0 and tr.status == RUNNING:
            if tr.pos >= self.corridor.stations[tr.loop_station].km + tr.cls.overlap_km - _EPS:
                self.loop_occ[tr.loop_station].remove(tr.tid)
                tr.loop_station = -1

    def _cross_boundary(self, tr: TrainState, t: float) -> bool:
        """Handle a train whose head is at the end of its block. Returns True if it entered the next block."""
        c = self.corridor
        k = tr.block + 1
        if k == c.n_blocks:
            self.occ[tr.block] = -1
            if tr.tail_block >= 0:
                self.occ[tr.tail_block] = -1
                tr.tail_block = -1
            tr.status = EXITED
            tr.exit_min = t
            if self.record:
                self.log.events.append([tr.tid, round(t, 3), round(tr.pos, 3), "exit"])
            return False
        st = self._boundary_station.get(k)
        at_station = st is not None and tr.next_stop == st
        if at_station:
            if tr.route_loop == st:
                tr.arrivals[st] = t
                tr.next_stop = st + 1
                tr.status = IN_LOOP
                tr.loop_station = st
                tr.route_loop = -1
                self.loop_reserved[st].remove(tr.tid)
                self.loop_occ[st].append(tr.tid)
                self.loop_uses[st] += 1
                tr.loop_clear_left = tr.cls.overlap_km / self.v_turn
                if self.record:
                    self.log.events.append([tr.tid, round(t, 3), round(tr.pos, 3), "loop_in"])
                return False
            if tr.cls.is_passenger and tr.cls.dwell_min > 0:
                tr.arrivals[st] = t
                tr.next_stop = st + 1
                tr.status = DWELL
                tr.dwell_left = tr.cls.dwell_min
                if self.record:
                    self.log.events.append([tr.tid, round(t, 3), round(tr.pos, 3), "arr"])
                return True  # continue the movement loop: time is consumed by the dwell
        if tr.hold:
            return False
        nb = tr.block + 1
        if self.occ[nb] != -1:
            return False
        if at_station:  # run-through: arrival = departure
            tr.arrivals[st] = t
            tr.next_stop = st + 1
        self.occ[nb] = tr.tid
        if tr.tail_block >= 0:
            self.occ[tr.tail_block] = -1
        tr.tail_block = tr.block
        tr.block = nb
        if st is not None and tr.departures[st] < 0:
            tr.departures[st] = t
            if self.record:
                self.log.events.append([tr.tid, round(t, 3), round(tr.pos, 3), "dep"])
        self._release_rear(tr)
        return True

    def _loop_tick(self, tr: TrainState, dt: float, t0: float) -> None:
        """A freight standing in a loop: clear the turnout, then depart when no passenger conflict remains."""
        st = tr.loop_station
        if tr.loop_clear_left > 0:
            tr.loop_clear_left -= dt
            if tr.loop_clear_left <= _EPS:
                tr.loop_clear_left = 0.0
                if self.occ[tr.block] == tr.tid:
                    self.occ[tr.block] = -1
                if tr.tail_block >= 0:
                    self.occ[tr.tail_block] = -1
                    tr.tail_block = -1
        tr.speed_now = 0.0
        tr.stopped = True
        if tr.loop_clear_left <= 0:
            nb = self.corridor.stations[st].boundary  # first block beyond the station
            if self.occ[nb] == -1 and (tr.release or not self._loop_conflict(tr, st)):
                self.occ[nb] = tr.tid
                tr.block = nb
                tr.tail_block = -1
                tr.status = RUNNING
                tr.from_loop_until_km = self.corridor.stations[st].km + self.corridor.turnout_zone_km
                tr.departures[st] = t0
                if self.record:
                    self.log.events.append([tr.tid, round(t0, 3), round(tr.pos, 3), "loop_out"])
        # standing in the loop (or departing: movement starts next substep) costs the full substep
        self._add_delay(tr, dt)

    def _loop_conflict(self, tr: TrainState, st: int) -> bool:
        """Conflict status of a looped freight, evaluated once per epoch (decision granularity)."""
        key = (tr.tid, self.epoch)
        hit = self._loop_conflict_cache.get(key)
        if hit is None:
            hit = bool(self.conflicts(tr, st))
            self._loop_conflict_cache = {key: hit} if len(self._loop_conflict_cache) > 64 else self._loop_conflict_cache
            self._loop_conflict_cache[key] = hit
        return hit

    # ------------------------------------------------------------ outputs
    def _record_frame(self) -> None:
        rows = []
        for tr in self.active:
            rows.append([tr.tid, round(tr.pos, 2), tr.status, round(tr.accrued_epoch, 2), round(tr.accrued, 1)])
        self.log.frames.append({"epoch": self.epoch, "t": round(self.now, 2), "trains": rows})

    def metrics(self) -> dict[str, Any]:
        """Episode metrics (valid at any time, final when ``done``)."""
        n_st = self.corridor.n_stations
        p_delay = f_delay = p_weighted = st_arr_delay = 0.0
        sec_p = [0.0] * (n_st + 1)
        sec_f = [0.0] * (n_st + 1)
        cleared = cleared_p = cleared_f = 0
        for tr in self.trains:
            if tr.cls.is_passenger:
                p_delay += tr.accrued
                p_weighted += tr.accrued * tr.cls.priority
                for i, a in enumerate(tr.arrivals):
                    if a >= 0:
                        st_arr_delay += max(0.0, a - tr.sched.sched_station_min[i])
                for s in range(n_st + 1):
                    sec_p[s] += tr.accrued_by_section[s]
            else:
                f_delay += tr.accrued
                for s in range(n_st + 1):
                    sec_f[s] += tr.accrued_by_section[s]
            if tr.status == EXITED:
                cleared += 1
                if tr.cls.is_passenger:
                    cleared_p += 1
                else:
                    cleared_f += 1
        decisions = sum(self.decision_count)
        invalid = sum(self.invalid_count)
        return {
            "passenger_delay": p_delay,
            "freight_delay": f_delay,
            "passenger_delay_weighted": p_weighted,
            "station_arrival_delay": st_arr_delay,
            "trains_cleared": cleared,
            "passengers_cleared": cleared_p,
            "freights_cleared": cleared_f,
            "n_trains": len(self.trains),
            "loop_uses": sum(self.loop_uses),
            "loop_uses_by_station": list(self.loop_uses),
            "decisions": decisions,
            "invalid_actions": invalid,
            "invalid_rate": invalid / decisions if decisions else 0.0,
            "action_counts": [list(r) for r in self.action_count],
            "section_passenger_delay": sec_p,
            "section_freight_delay": sec_f,
            "shared_return": self.total_shared_return,
            "epochs": self.epoch,
        }
