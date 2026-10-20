"""Map raw local observations to a tabular state index.

Features (bin edges from ``config.yaml -> discretization``):

====================  ==========================================  =======
feature               bins                                        levels
====================  ==========================================  =======
passenger ETA (min)   eta_edges_min                               len+1
freight position      0 = standing in loop, then dist_edges_km    len+2
freight speed         speed_edges_kmh (stopped / moving)          len+1
braking margin        margin_edges_km (no room / room)            len+1
loop busy             0 / 1                                       2
local delay (min)     local_delay_edges_min                       len+1
downstream signal     congestion level 0/1/2 (1 level if ablated) 3 or 1
====================  ==========================================  =======

The index is a mixed-radix number over these features.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from sim.env import Observation


@dataclass(frozen=True)
class Discretizer:
    """Mixed-radix tabular encoder for :class:`Observation`."""

    eta_edges: tuple[float, ...]
    dist_edges: tuple[float, ...]
    speed_edges: tuple[float, ...]
    margin_edges: tuple[float, ...]
    local_edges: tuple[float, ...]
    congestion_levels: int
    use_signal: bool

    @property
    def radices(self) -> tuple[int, ...]:
        """Number of levels for each feature, in encoding order."""
        return (
            len(self.eta_edges) + 1,
            len(self.dist_edges) + 2,
            len(self.speed_edges) + 1,
            len(self.margin_edges) + 1,
            2,
            len(self.local_edges) + 1,
            self.congestion_levels if self.use_signal else 1,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names in encoding order."""
        return ("eta", "position", "speed", "margin", "loop_busy", "local_delay", "downstream")

    @property
    def n_states(self) -> int:
        """Size of the tabular state space per agent."""
        n = 1
        for r in self.radices:
            n *= r
        return n

    def features(self, obs: Observation) -> tuple[int, ...]:
        """Discrete feature levels for an observation."""
        pos = 0 if obs.in_loop else 1 + bisect_right(self.dist_edges, obs.dist_km)
        return (
            bisect_right(self.eta_edges, obs.eta_min),
            pos,
            bisect_right(self.speed_edges, obs.speed_kmh),
            bisect_right(self.margin_edges, obs.margin_km + 1e-9),
            int(obs.loop_busy),
            bisect_right(self.local_edges, obs.local_delay_min),
            obs.downstream if self.use_signal else 0,
        )

    def encode(self, obs: Observation) -> int:
        """Tabular state index for an observation."""
        idx = 0
        for level, radix in zip(self.features(obs), self.radices):
            idx = idx * radix + min(level, radix - 1)
        return idx

    def level_labels(self) -> dict[str, list[str]]:
        """Human-readable label of every level of every feature, derived from the bin edges (for the UI)."""

        def bins(edges: tuple[float, ...], unit: str) -> list[str]:
            e = [f"{x:g}" for x in edges]
            return [f"<{e[0]}{unit}"] + [f"{a}–{b}{unit}" for a, b in zip(e, e[1:])] + [f"≥{e[-1]}{unit}"]

        return {
            "eta": bins(self.eta_edges, " min"),
            "position": ["in loop"] + bins(self.dist_edges, " km"),
            "speed": ["stopped", "moving"] if len(self.speed_edges) == 1 else bins(self.speed_edges, " km/h"),
            "margin": ["no room", "room"] if len(self.margin_edges) == 1 else bins(self.margin_edges, " km"),
            "loop_busy": ["free", "busy"],
            "local_delay": bins(self.local_edges, " min"),
            "downstream": (["low", "mid", "high"] if self.congestion_levels == 3
                           else [str(i) for i in range(self.congestion_levels)]) if self.use_signal else ["(off)"],
        }

    def decode(self, idx: int) -> dict[str, int]:
        """Inverse of :meth:`encode` (feature name -> level)."""
        levels: list[int] = []
        for radix in reversed(self.radices):
            levels.append(idx % radix)
            idx //= radix
        return dict(zip(self.feature_names, reversed(levels)))

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description of the bins (exported for the Method page)."""
        return {
            "features": list(self.feature_names),
            "radices": list(self.radices),
            "n_states": self.n_states,
            "eta_edges_min": list(self.eta_edges),
            "dist_edges_km": list(self.dist_edges),
            "speed_edges_kmh": list(self.speed_edges),
            "margin_edges_km": list(self.margin_edges),
            "local_delay_edges_min": list(self.local_edges),
            "congestion_levels": self.congestion_levels,
            "use_downstream_signal": self.use_signal,
            "level_labels": self.level_labels(),
        }


def build_discretizer(cfg: dict[str, Any]) -> Discretizer:
    """Build the discretizer from ``config['discretization']``."""
    d = cfg["discretization"]
    return Discretizer(
        eta_edges=tuple(float(x) for x in d["eta_edges_min"]),
        dist_edges=tuple(float(x) for x in d["dist_edges_km"]),
        speed_edges=tuple(float(x) for x in d["speed_edges_kmh"]),
        margin_edges=tuple(float(x) for x in d["margin_edges_km"]),
        local_edges=tuple(float(x) for x in d["local_delay_edges_min"]),
        congestion_levels=len(d["congestion_edges"]) + 1,
        use_signal=bool(d["use_downstream_signal"]),
    )
