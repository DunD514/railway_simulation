"""Train classes and per-train mutable state used by the simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Train status codes (kept as small ints so replays stay compact).
WAITING, RUNNING, DWELL, IN_LOOP, EXITED = 0, 1, 2, 3, 4
STATUS_NAMES = {WAITING: "waiting", RUNNING: "running", DWELL: "dwell", IN_LOOP: "loop", EXITED: "exited"}


@dataclass(frozen=True)
class TrainClass:
    """Static properties of a rolling-stock / service class."""

    code: str
    name: str
    kind: str            # "passenger" | "freight"
    speed_kmh: float
    priority: float
    overlap_km: float    # extra clearance distance before the block in rear is released
    dwell_min: float     # scheduled stop at each station (passengers only)
    is_passenger: bool   # derived from kind (stored, not computed, because the sim reads it constantly)
    speed_kpm: float     # line speed in km per minute


def load_train_classes(cfg: dict[str, Any]) -> dict[str, TrainClass]:
    """Build the train classes from config."""
    out: dict[str, TrainClass] = {}
    for code, c in cfg["trains"]["classes"].items():
        out[code] = TrainClass(
            code=code,
            name=str(c["name"]),
            kind=str(c["kind"]),
            speed_kmh=float(c["speed_kmh"]),
            priority=float(c["priority"]),
            overlap_km=float(c["overlap_km"]),
            dwell_min=float(c["dwell_min"]),
            is_passenger=str(c["kind"]) == "passenger",
            speed_kpm=float(c["speed_kmh"]) / 60.0,
        )
    return out


@dataclass(frozen=True)
class ScheduledTrain:
    """One timetable entry (immutable)."""

    tid: int
    cls: TrainClass
    sched_entry_min: float      # scheduled time at the corridor entry (km 0)
    ready_min: float            # actual time the train is ready to enter (schedule + jitter + initial delay)
    sched_station_min: tuple[float, ...]  # scheduled arrival at each station

    @property
    def label(self) -> str:
        """Human-readable train id, e.g. 'RAJ-07'."""
        return f"{self.cls.code}-{self.tid:02d}"


class TrainState:
    """Mutable runtime state of one train inside an episode."""

    __slots__ = (
        "sched", "cls", "tid", "pos", "status", "block", "tail_block", "loop_station", "route_loop",
        "hold", "hold_streak", "release", "dwell_left", "dwell_station", "from_loop_until_km", "loop_clear_left", "accrued",
        "accrued_epoch", "entered_min", "exit_min", "arrivals", "departures", "next_stop", "stopped",
        "speed_now", "accrued_by_section",
    )

    def __init__(self, sched: ScheduledTrain, n_stations: int) -> None:
        self.sched = sched
        self.cls = sched.cls
        self.tid = sched.tid
        self.pos = 0.0
        self.status = WAITING
        self.block = -1
        self.tail_block = -1            # block in rear still held until the overlap is cleared
        self.loop_station = -1          # station whose loop the train is standing in
        self.route_loop = -1            # station whose loop route has been set for this train
        self.hold = False               # HOLD: do not enter a new block this epoch
        self.hold_streak = 0            # consecutive epochs this freight has been held
        self.release = False            # agent ordered a looped freight to depart this epoch
        self.dwell_left = 0.0
        self.dwell_station = -1
        self.from_loop_until_km = -1.0  # turnout speed restriction after leaving a loop
        self.loop_clear_left = 0.0      # minutes until the train has cleared the loop turnout
        self.accrued = 0.0              # delay-minutes accrued inside the corridor
        self.accrued_epoch = 0.0
        self.entered_min = -1.0
        self.exit_min = -1.0
        self.arrivals = [-1.0] * n_stations
        self.departures = [-1.0] * n_stations
        self.next_stop = 0              # next station index not yet reached
        self.stopped = False            # did not move during the last substep
        self.speed_now = 0.0            # km/h during last substep
        self.accrued_by_section = [0.0] * (n_stations + 1)
