import { useMemo, useState } from "react";
import { clock, type Json } from "../api";
import { CLASS_COLORS } from "./StringDiagram";

const STATUS_NAMES: Record<number, { name: string; color: string; desc: string }> = {
  0: { name: "Waiting", color: "#8b9097", desc: "Waiting at corridor origin to enter track" },
  1: { name: "Cruising", color: "#3d7a5c", desc: "Running on mainline track at scheduled line speed" },
  2: { name: "Station Stop", color: "#2b5f97", desc: "Dwelling at station platform for passenger embarkation" },
  3: { name: "In Siding Loop", color: "#c99a2e", desc: "Diverted into station siding track to allow faster express to overtake" },
  4: { name: "Exited", color: "#6e6556", desc: "Successfully completed corridor run and cleared exit" },
};

interface InterpolatedTrain {
  tid: number;
  pos: number;
  status: number;
  accrued: number;
  speed: number;
  tr: Json;
  isInLoop: boolean;
  isDwelling: boolean;
  isHolding: boolean;
}

export function CorridorLiveMap({
  replay,
  stations,
  lengthKm,
  t,
  onSeek: _onSeek,
  focus,
  onSelectTrain,
}: {
  replay: Json;
  stations: { code: string; name: string; km: number }[];
  lengthKm: number;
  t: number;
  onSeek: (time: number) => void;
  focus: number | null;
  onSelectTrain: (tid: number | null) => void;
}) {
  const [selectedStation, setSelectedStation] = useState<string | null>(null);
  const [filterKind, setFilterKind] = useState<"all" | "passenger" | "freight" | "loop">("all");

  const byId = useMemo(() => new Map<number, Json>(replay.trains.map((tr: Json) => [tr.tid, tr])), [replay]);

  // Interpolate train positions at time t from adjacent recording frames
  const { activeTrains, loopOccupancy } = useMemo(() => {
    const fs = replay.frames;
    if (!fs || fs.length === 0) return { activeTrains: [], loopOccupancy: new Map() };

    let lo = 0;
    let hi = fs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (fs[mid].t <= t) lo = mid;
      else hi = mid - 1;
    }
    const f0 = fs[lo];
    const f1 = fs[Math.min(fs.length - 1, lo + 1)];
    const ratio = f1 && f1.t > f0.t ? Math.max(0, Math.min(1, (t - f0.t) / (f1.t - f0.t))) : 0;

    const tMap1 = new Map<number, number[]>();
    if (f1 && f1.trains) {
      for (const row of f1.trains) tMap1.set(row[0], row);
    }

    const result: InterpolatedTrain[] = [];
    const occ = new Map<string, InterpolatedTrain>();

    if (f0 && f0.trains) {
      for (const row0 of f0.trains) {
        const tid = row0[0];
        const tr = byId.get(tid);
        if (!tr) continue;

        const row1 = tMap1.get(tid);
        let pos = row0[1];
        let status = row0[2];
        let accrued = row0[4] ?? 0;

        if (row1) {
          pos = pos + ratio * (row1[1] - pos);
          accrued = accrued + ratio * ((row1[4] ?? accrued) - accrued);
          if (row1[2] === 3 || row0[2] === 3) {
            status = row0[2] === 3 ? 3 : row1[2];
          }
        }

        const isInLoop = status === 3;
        const isDwelling = status === 2;
        const isHolding = Boolean(status === 1 && row1 && Math.abs(row1[1] - row0[1]) < 0.05 && tr.speed_kmh > 0);

        const trainObj: InterpolatedTrain = {
          tid,
          pos,
          status,
          accrued,
          speed: isInLoop || isDwelling ? 0 : tr.speed_kmh,
          tr,
          isInLoop,
          isDwelling,
          isHolding,
        };

        result.push(trainObj);

        if (isInLoop) {
          // Identify which station loop
          const nearStation = stations.find((s) => Math.abs(s.km - pos) <= 15);
          if (nearStation) occ.set(nearStation.code, trainObj);
        }
      }
    }

    return { activeTrains: result, loopOccupancy: occ };
  }, [replay, t, byId, stations]);

  // Find active overtakes: a passenger train cruising near a freight in siding loop
  const activeOvertakes = useMemo(() => {
    const list: { passenger: InterpolatedTrain; freight: InterpolatedTrain; station: { code: string; km: number } }[] = [];
    for (const [stCode, freight] of loopOccupancy.entries()) {
      const st = stations.find((s) => s.code === stCode);
      if (!st) continue;
      const passingPax = activeTrains.find(
        (tr) => tr.tr.kind === "passenger" && !tr.isInLoop && Math.abs(tr.pos - st.km) <= 18
      );
      if (passingPax) {
        list.push({ passenger: passingPax, freight, station: st });
      }
    }
    return list;
  }, [loopOccupancy, activeTrains, stations]);

  // Latest decision banner around time t
  const latestDecision = useMemo(() => {
    if (!replay.decisions) return null;
    const past = replay.decisions.filter((d: Json) => d.t <= t);
    return past.length > 0 ? past[past.length - 1] : null;
  }, [replay.decisions, t]);

  const focusedTrain = useMemo(() => {
    if (focus === null) return null;
    return activeTrains.find((tr) => tr.tid === focus) ?? byId.get(focus) ?? null;
  }, [focus, activeTrains, byId]);

  // Filtered train list
  const filteredTrains = useMemo(() => {
    return activeTrains.filter((tr) => {
      if (filterKind === "passenger") return tr.tr.kind === "passenger";
      if (filterKind === "freight") return tr.tr.kind === "freight";
      if (filterKind === "loop") return tr.isInLoop;
      return true;
    });
  }, [activeTrains, filterKind]);

  // Coordinate projection
  const viewWidth = 1000;
  const viewHeight = 280;
  const paddingX = 40;
  const trackY = 150;
  const loopY = 85;

  const toX = (km: number) => {
    return paddingX + (km / lengthKm) * (viewWidth - 2 * paddingX);
  };

  return (
    <div className="live-corridor-wrapper">
      {/* Top Banner with live operational context */}
      <div className="live-corridor-header">
        <div className="live-badge-group">
          <span className="live-indicator-dot"></span>
          <span className="live-title">LIVE CORRIDOR RADAR</span>
          <span className="live-time-chip">⏱ {clock(t)}</span>
          <span className="live-stat-chip">
            <b>{activeTrains.length}</b> Trains Active (
            {activeTrains.filter((tr) => tr.tr.kind === "passenger").length} Pax /{" "}
            {activeTrains.filter((tr) => tr.tr.kind === "freight").length} Frt)
          </span>
          {loopOccupancy.size > 0 && (
            <span className="live-loop-chip">
              ⚠️ <b>{loopOccupancy.size}</b> in Station Siding Loop
            </span>
          )}
        </div>

        {/* Filter / View Toggles */}
        <div className="live-filter-group">
          <div className="seg">
            <button className={filterKind === "all" ? "on" : ""} onClick={() => setFilterKind("all")}>
              All Trains ({activeTrains.length})
            </button>
            <button className={filterKind === "passenger" ? "on" : ""} onClick={() => setFilterKind("passenger")}>
              Passenger (
              {activeTrains.filter((tr) => tr.tr.kind === "passenger").length})
            </button>
            <button className={filterKind === "freight" ? "on" : ""} onClick={() => setFilterKind("freight")}>
              Freight ({activeTrains.filter((tr) => tr.tr.kind === "freight").length})
            </button>
            <button className={filterKind === "loop" ? "on" : ""} onClick={() => setFilterKind("loop")}>
              In Loop ({loopOccupancy.size})
            </button>
          </div>
        </div>
      </div>

      {/* Latest Dispatch Action Alert (Layperson English) */}
      {latestDecision && (
        <div className={`live-dispatch-alert action-${latestDecision.action}`}>
          <div className="dispatch-alert-icon">
            {latestDecision.action === 1 ? "🔀" : latestDecision.action === 2 ? "🛑" : "🟢"}
          </div>
          <div className="dispatch-alert-text">
            <strong>
              {clock(latestDecision.t)} · Station {stations[latestDecision.agent]?.code} (
              {stations[latestDecision.agent]?.name})
            </strong>
            :{" "}
            {latestDecision.action === 1 ? (
              <span>
                AI directed Freight <b>{byId.get(latestDecision.freight)?.label}</b> into the{" "}
                <b style={{ color: "var(--c-loop)" }}>Siding Loop</b> to allow high-speed express train (ETA{" "}
                {latestDecision.eta.toFixed(0)}m) to overtake cleanly.
              </span>
            ) : latestDecision.action === 2 ? (
              <span>
                AI held Freight <b>{byId.get(latestDecision.freight)?.label}</b> at station approach signal to manage
                headway.
              </span>
            ) : (
              <span>
                AI cleared Freight <b>{byId.get(latestDecision.freight)?.label}</b> to <b>PROCEED</b> along the mainline
                without stopping.
              </span>
            )}
          </div>
        </div>
      )}

      {/* Interactive Map Visual */}
      <div className="live-map-canvas-container">
        <svg viewBox={`0 0 ${viewWidth} ${viewHeight}`} className="live-map-svg" preserveAspectRatio="xMidYMid meet">
          <defs>
            {/* Gradients for tracks and trains */}
            <linearGradient id="mainline-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#3d4957" />
              <stop offset="100%" stopColor="#1f2c3d" />
            </linearGradient>
            <linearGradient id="loop-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#e39a2b" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#c27d19" stopOpacity="0.8" />
            </linearGradient>
            <filter id="glow-gold" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#f5a623" floodOpacity="0.6" />
            </filter>
            <filter id="glow-train" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="1" stdDeviation="3" floodColor="rgba(0,0,0,0.35)" />
            </filter>
          </defs>

          {/* Background corridor base plate */}
          <rect x={10} y={15} width={viewWidth - 20} height={viewHeight - 30} rx={16} fill="var(--surface-2)" />

          {/* Distance grid ticks along the corridor */}
          {[0, 100, 200, 300, 400, 500, 600, 700, 800, 870].map((km) => (
            <g key={km} transform={`translate(${toX(km)}, 0)`}>
              <line x1={0} y1={trackY - 70} x2={0} y2={trackY + 45} stroke="var(--border)" strokeDasharray="3 3" opacity={0.6} />
              <text x={0} y={trackY + 58} textAnchor="middle" className="live-km-tick">
                {km} km
              </text>
            </g>
          ))}

          {/* Direction indicator (UP direction: Kanpur -> Asansol) */}
          <g transform={`translate(${viewWidth - 110}, 36)`}>
            <rect x={-5} y={-14} width={100} height={20} rx={6} fill="var(--surface)" stroke="var(--border)" />
            <text x={45} y={0} textAnchor="middle" fill="var(--muted)" fontSize={11} fontWeight={600}>
              EASTBOUND ➔
            </text>
          </g>

          {/* SIDING LOOPS at each of the 6 stations */}
          {stations.map((st) => {
            const stX = toX(st.km);
            const loopLen = 32;
            const x0 = stX - loopLen;
            const x1 = stX + loopLen;
            const isOccupied = loopOccupancy.has(st.code);
            const occupiedBy = loopOccupancy.get(st.code);

            return (
              <g key={`loop-${st.code}`} className="station-loop-assembly">
                {/* Turnout switch tracks (entering and exiting loop siding) */}
                <path
                  d={`M ${x0 - 18} ${trackY} C ${x0 - 6} ${trackY}, ${x0 - 8} ${loopY}, ${x0} ${loopY} L ${x1} ${loopY} C ${x1 + 8} ${loopY}, ${x1 + 6} ${trackY}, ${x1 + 18} ${trackY}`}
                  fill="none"
                  stroke={isOccupied ? "var(--c-loop)" : "var(--border-strong)"}
                  strokeWidth={isOccupied ? 3.5 : 2}
                  strokeDasharray={isOccupied ? "none" : "4 2"}
                />

                {/* Siding Loop Platform / Siding Track highlight */}
                <line
                  x1={x0}
                  y1={loopY}
                  x2={x1}
                  y2={loopY}
                  stroke={isOccupied ? "var(--c-loop)" : "var(--border-strong)"}
                  strokeWidth={4.5}
                  strokeLinecap="round"
                />

                {/* Siding Loop Status Tag */}
                <g transform={`translate(${stX}, ${loopY - 14})`}>
                  <rect
                    x={-42}
                    y={-10}
                    width={84}
                    height={16}
                    rx={8}
                    fill={isOccupied ? "#fef3c7" : "var(--surface)"}
                    stroke={isOccupied ? "#d97706" : "var(--border)"}
                    strokeWidth={1}
                  />
                  <text
                    x={0}
                    y={2}
                    textAnchor="middle"
                    fontSize={9.5}
                    fontWeight={700}
                    fill={isOccupied ? "#b45309" : "var(--muted)"}
                  >
                    {isOccupied ? `LOOP: ${occupiedBy?.tr.label}` : "LOOP EMPTY"}
                  </text>
                </g>
              </g>
            );
          })}

          {/* MAINLINE TRACK */}
          {/* Railway Ties / Sleepers pattern */}
          <line
            x1={toX(0)}
            y1={trackY}
            x2={toX(lengthKm)}
            y2={trackY}
            stroke="#94a3b8"
            strokeWidth={8}
            strokeDasharray="2 6"
            strokeLinecap="round"
            opacity={0.4}
          />
          {/* Steel Rails */}
          <line
            x1={toX(0)}
            y1={trackY - 2.5}
            x2={toX(lengthKm)}
            y2={trackY - 2.5}
            stroke="#334155"
            strokeWidth={2}
          />
          <line
            x1={toX(0)}
            y1={trackY + 2.5}
            x2={toX(lengthKm)}
            y2={trackY + 2.5}
            stroke="#334155"
            strokeWidth={2}
          />

          {/* STATIONS: Platforms, Signboards, Signals */}
          {stations.map((st) => {
            const stX = toX(st.km);
            const isOccupied = loopOccupancy.has(st.code);

            return (
              <g
                key={`st-${st.code}`}
                transform={`translate(${stX}, 0)`}
                className="station-node"
                onClick={() => setSelectedStation(selectedStation === st.code ? null : st.code)}
                style={{ cursor: "pointer" }}
              >
                {/* Station Platform Line & Canopy */}
                <rect x={-24} y={trackY - 10} width={48} height={20} rx={4} fill="#e2e8f0" stroke="#94a3b8" strokeWidth={1.5} opacity={0.7} />
                <line x1={-20} y1={trackY + 12} x2={20} y2={trackY + 12} stroke="#64748b" strokeWidth={3} strokeLinecap="round" />

                {/* 3-Aspect Signal Light at Station Approach */}
                <g transform={`translate(-34, ${trackY - 24})`}>
                  <rect x={-4} y={0} width={8} height={18} rx={4} fill="#1e293b" />
                  <circle cx={0} cy={5} r={2.5} fill={isOccupied ? "#ef4444" : "#22c55e"} />
                  <circle cx={0} cy={13} r={2.5} fill={isOccupied ? "#f59e0b" : "#475569"} />
                </g>

                {/* Station Name & Code Signboard (Indian Railways classic yellow style) */}
                <g transform={`translate(0, ${trackY + 36})`}>
                  <rect
                    x={-34}
                    y={-10}
                    width={68}
                    height={20}
                    rx={5}
                    fill="#fef08a"
                    stroke="#ca8a04"
                    strokeWidth={1.2}
                    filter="drop-shadow(0 1px 2px rgba(0,0,0,0.1))"
                  />
                  <text x={0} y={4} textAnchor="middle" fontSize={11} fontWeight={800} fill="#713f12" letterSpacing="0.03em">
                    {st.code}
                  </text>
                  <text x={0} y={18} textAnchor="middle" fontSize={9.5} fill="var(--text)" fontWeight={500}>
                    {st.name.split(" ")[0]}
                  </text>
                </g>
              </g>
            );
          })}

          {/* ACTIVE OVERTAKE HIGHLIGHT CONNECTIONS */}
          {activeOvertakes.map((ovt, idx) => {
            const paxX = toX(ovt.passenger.pos);
            const frtX = toX(ovt.freight.pos);
            return (
              <g key={`ovt-${idx}`} className="overtake-highlight-group">
                <path
                  d={`M ${paxX} ${trackY} Q ${(paxX + frtX) / 2} ${(trackY + loopY) / 2} ${frtX} ${loopY}`}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth={2.5}
                  strokeDasharray="4 4"
                />
                <g transform={`translate(${(paxX + frtX) / 2}, ${(trackY + loopY) / 2})`}>
                  <rect x={-55} y={-11} width={110} height={22} rx={11} fill="#f59e0b" filter="url(#glow-gold)" />
                  <text x={0} y={4} textAnchor="middle" fill="#fff" fontSize={10} fontWeight={800}>
                    ⚡ OVERTAKE IN PROGRESS
                  </text>
                </g>
              </g>
            );
          })}

          {/* TRAINS RUNNING ON TRACKS */}
          {filteredTrains.map((tr) => {
            const trX = toX(tr.pos);
            const trY = tr.isInLoop ? loopY : trackY;
            const isFocused = focus === tr.tid;
            const color = CLASS_COLORS[tr.tr.cls] ?? "var(--c-frt)";
            const isPax = tr.tr.kind === "passenger";

            return (
              <g
                key={`train-${tr.tid}`}
                transform={`translate(${trX}, ${trY})`}
                className={`train-avatar-group ${isFocused ? "focused" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectTrain(isFocused ? null : tr.tid);
                }}
                style={{ cursor: "pointer" }}
              >
                {/* Pulsing halo if selected */}
                {isFocused && (
                  <circle cx={0} cy={0} r={22} fill="none" stroke="var(--accent)" strokeWidth={2.5} strokeDasharray="3 3" />
                )}

                {/* Train Body Graphics */}
                {isPax ? (
                  // Aerodynamic Express Passenger Train
                  <g filter="url(#glow-train)">
                    {/* Headlight beam */}
                    <polygon points="14,0 35,-6 35,6" fill="rgba(254, 240, 138, 0.45)" />
                    {/* Main Locomotive Coach */}
                    <path
                      d="M -16,-6 L 10,-6 C 16,-6, 18,-2, 18,0 C 18,2, 16,6, 10,6 L -16,6 Z"
                      fill={color}
                      stroke="#fff"
                      strokeWidth={1.5}
                    />
                    {/* Windshield */}
                    <path d="M 6,-4 L 13,-4 C 15,-4, 15,4, 13,4 L 6,4 Z" fill="#38bdf8" />
                    {/* Trailing passenger coaches */}
                    <rect x={-32} y={-5} width={13} height={10} rx={2} fill={color} stroke="#fff" strokeWidth={1} />
                    <rect x={-47} y={-5} width={13} height={10} rx={2} fill={color} stroke="#fff" strokeWidth={1} />
                  </g>
                ) : (
                  // Heavy Freight Train with Wagons
                  <g filter="url(#glow-train)">
                    {/* Diesel Locomotive */}
                    <rect x={-8} y={-7} width={20} height={14} rx={3} fill="#ea580c" stroke="#fff" strokeWidth={1.5} />
                    <rect x={-2} y={-5} width={6} height={10} fill="#fef08a" />
                    {/* Freight Container Wagons */}
                    <rect x={-24} y={-6} width={13} height={12} rx={1.5} fill="#475569" stroke="#94a3b8" strokeWidth={1} />
                    <rect x={-39} y={-6} width={13} height={12} rx={1.5} fill="#b45309" stroke="#fed7aa" strokeWidth={1} />
                    <rect x={-54} y={-6} width={13} height={12} rx={1.5} fill="#475569" stroke="#94a3b8" strokeWidth={1} />
                  </g>
                )}

                {/* Train Label Badge */}
                <g transform={`translate(0, ${tr.isInLoop ? -14 : -16})`}>
                  <rect
                    x={-28}
                    y={-10}
                    width={56}
                    height={18}
                    rx={4}
                    fill="var(--surface)"
                    stroke={color}
                    strokeWidth={1.8}
                    filter="drop-shadow(0 1px 3px rgba(0,0,0,0.15))"
                  />
                  <text x={0} y={3} textAnchor="middle" fontSize={10} fontWeight={800} fill="var(--text)">
                    {tr.tr.label}
                  </text>
                </g>

                {/* Speed / Status Pill below train */}
                <g transform={`translate(0, 16)`}>
                  <rect
                    x={-24}
                    y={-7}
                    width={48}
                    height={14}
                    rx={3}
                    fill={tr.isInLoop ? "#fef3c7" : "rgba(15, 23, 42, 0.75)"}
                  />
                  <text
                    x={0}
                    y={3.5}
                    textAnchor="middle"
                    fontSize={8.5}
                    fontWeight={700}
                    fill={tr.isInLoop ? "#92400e" : "#fff"}
                  >
                    {tr.isInLoop ? "WAITING" : `${Math.round(tr.speed)} km/h`}
                  </text>
                </g>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Floating / Integrated Train Inspector Card */}
      {focusedTrain && (
        <div className="train-inspector-card">
          <div className="inspector-head">
            <div className="inspector-title">
              <span className="swatch" style={{ background: CLASS_COLORS[focusedTrain.tr?.cls ?? focusedTrain.cls] }} />
              <div>
                <h3>{focusedTrain.tr?.label ?? focusedTrain.label}</h3>
                <span className="muted small">
                  {focusedTrain.tr?.kind === "passenger" ? "Passenger Express" : "Heavy Freight Cargo"} ·{" "}
                  {focusedTrain.tr?.cls ?? focusedTrain.cls} ({focusedTrain.tr?.speed_kmh ?? focusedTrain.speed_kmh} km/h Line Speed)
                </span>
              </div>
            </div>
            <button className="btn small" onClick={() => onSelectTrain(null)}>
              ✕ Close
            </button>
          </div>

          <div className="inspector-grid">
            <div className="inspector-metric">
              <span className="label">Current Status</span>
              <span className="value status-badge" style={{ color: STATUS_NAMES[focusedTrain.status ?? 1]?.color }}>
                ● {STATUS_NAMES[focusedTrain.status ?? 1]?.name ?? "Running"}
              </span>
              <small className="muted">{STATUS_NAMES[focusedTrain.status ?? 1]?.desc}</small>
            </div>

            <div className="inspector-metric">
              <span className="label">Corridor Position</span>
              <span className="value">
                {focusedTrain.pos !== undefined ? `${focusedTrain.pos.toFixed(1)} km` : "At Origin"}
              </span>
              <small className="muted">
                {focusedTrain.pos !== undefined
                  ? stations.reduce((prev, curr) => (curr.km <= focusedTrain.pos ? curr : prev), stations[0])?.name
                  : "Entry stretch"}
              </small>
            </div>

            <div className="inspector-metric">
              <span className="label">Current Delay</span>
              <span className={`value ${(focusedTrain.accrued ?? focusedTrain.delay ?? 0) > 10 ? "warn" : "good"}`}>
                {(focusedTrain.accrued ?? focusedTrain.delay ?? 0).toFixed(0)} min
              </span>
              <small className="muted">
                {(focusedTrain.accrued ?? focusedTrain.delay ?? 0) <= 5
                  ? "🟢 On schedule"
                  : (focusedTrain.accrued ?? focusedTrain.delay ?? 0) <= 20
                  ? "🟡 Minor delay absorbed"
                  : "🔴 Significant wait time"}
              </small>
            </div>

            <div className="inspector-metric">
              <span className="label">AI Mission Strategy</span>
              <span className="value" style={{ fontSize: 13 }}>
                {focusedTrain.isInLoop
                  ? "Holding in loop siding so faster express can overtake"
                  : focusedTrain.tr?.kind === "passenger"
                  ? "High-priority clearance across all 6 junctions"
                  : "Moving eastward on clear headway"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
