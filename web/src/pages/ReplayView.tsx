import { useEffect, useMemo, useRef, useState } from "react";
import { ACTION_NAMES, clock, fmt, type Json } from "../api";
import { CorridorLiveMap } from "../charts/CorridorLiveMap";
import { LaypersonStats } from "../charts/LaypersonStats";
import { Legend } from "../charts/common";
import { CLASS_COLORS, StringDiagram } from "../charts/StringDiagram";

type ViewMode = "map" | "stats" | "chart" | "combined";

/** Player, live radar track map, plain-English statistics, string diagram, and train list. */
export function ReplayView({ rp, ov }: { rp: Json; ov: Json }) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState<number>(20); // sim minutes per real second
  const [focus, setFocus] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("map");
  const raf = useRef<number>(0);

  const tMax: number = rp.frames?.at(-1)?.t ?? 1440;

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const step = (now: number) => {
      setT((x) => {
        const nx = x + ((now - last) / 1000) * playSpeed;
        if (nx >= tMax) {
          setPlaying(false);
          return tMax;
        }
        return nx;
      });
      last = now;
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, tMax, playSpeed]);

  const stations = (ov.stations as { code: string; name: string; km: number }[]) ?? [];
  const byId = useMemo(() => new Map<number, Json>(rp.trains.map((tr: Json) => [tr.tid, tr])), [rp]);

  // Jump to next loop or overtake event
  const jumpToNextEvent = () => {
    const loopEvents = (rp.events ?? [])
      .filter((ev: Json[]) => (ev[3] === "loop_in" || ev[3] === "loop_out") && ev[1] > t + 1)
      .sort((a: Json[], b: Json[]) => a[1] - b[1]);

    if (loopEvents.length > 0) {
      setT(Math.floor(loopEvents[0][1]));
    } else {
      // Loop over to start or first event
      setT(0);
    }
  };

  const recent = useMemo(
    () => (rp.decisions ?? []).filter((d: Json) => d.t <= t && d.t > t - 30 && !d.in_loop).slice(-8).reverse(),
    [rp, t],
  );

  return (
    <>
      <section className="card">
        {/* Navigation / View Mode Switcher Header */}
        <div className="card-head" style={{ marginBottom: 12 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <h2 style={{ fontSize: 18 }}>Corridor Interactive Monitor</h2>
              <span className="live-clock-badge">⏱ {clock(t)}</span>
            </div>
            <span className="small muted">
              Kanpur Central &rarr; Asansol Jn (870 km corridor · 6 station precedence junctions)
            </span>
          </div>

          {/* View Mode Segmented Controls */}
          <div className="seg">
            <button className={viewMode === "map" ? "on" : ""} onClick={() => setViewMode("map")}>
              🗺️ Live Track Map
            </button>
            <button className={viewMode === "stats" ? "on" : ""} onClick={() => setViewMode("stats")}>
              📊 Meaningful Statistics
            </button>
            <button className={viewMode === "chart" ? "on" : ""} onClick={() => setViewMode("chart")}>
              📈 Time–Distance Diagram
            </button>
            <button className={viewMode === "combined" ? "on" : ""} onClick={() => setViewMode("combined")}>
              🔀 Map &amp; Diagram
            </button>
          </div>
        </div>

        {/* Unified Playback Bar */}
        <div className="player" style={{ marginBottom: 16 }}>
          <button
            className="play"
            aria-label={playing ? "Pause" : "Play"}
            onClick={() => {
              if (t >= tMax) setT(0);
              setPlaying(!playing);
            }}
          >
            <svg viewBox="0 0 12 12" width="14" height="14" aria-hidden>
              {playing ? (
                <path d="M2 1h3v10H2zM7 1h3v10H7z" fill="currentColor" />
              ) : (
                <path d="M2 1l9 5-9 5z" fill="currentColor" />
              )}
            </svg>
          </button>

          {/* Step Back button */}
          <button
            type="button"
            className="btn small"
            title="Step back 15 minutes"
            onClick={() => setT(Math.max(0, t - 15))}
          >
            -15m
          </button>

          {/* Time Scrubber */}
          <input
            type="range"
            min={0}
            max={tMax}
            step={1}
            value={t}
            onChange={(e) => {
              setT(+e.target.value);
              setPlaying(false);
            }}
          />

          {/* Step Forward button */}
          <button
            type="button"
            className="btn small"
            title="Step forward 15 minutes"
            onClick={() => setT(Math.min(tMax, t + 15))}
          >
            +15m
          </button>

          {/* Jump to Next Overtake Event */}
          <button
            type="button"
            className="btn small"
            style={{ fontWeight: 600, background: "var(--surface-2)" }}
            title="Jump forward to the next loop overtake event"
            onClick={jumpToNextEvent}
          >
            ⚡ Next Overtake
          </button>

          <span className="clock" style={{ fontSize: 16, minWidth: 60 }}>
            {clock(t)}
          </span>

          {/* Speed Selector */}
          <div className="speed-pills">
            {[5, 10, 20, 40].map((s) => (
              <button
                key={s}
                type="button"
                className={`speed-pill ${playSpeed === s ? "active" : ""}`}
                onClick={() => setPlaySpeed(s)}
              >
                {s === 5 ? "0.5x" : s === 10 ? "1x" : s === 20 ? "2x" : "4x"}
              </button>
            ))}
          </div>
        </div>

        {/* View 1: Real-Time Interactive Track Map */}
        {(viewMode === "map" || viewMode === "combined") && (
          <div style={{ marginBottom: viewMode === "combined" ? 24 : 0 }}>
            <CorridorLiveMap
              replay={rp}
              stations={stations}
              lengthKm={ov.length_km}
              t={t}
              onSeek={setT}
              focus={focus}
              onSelectTrain={setFocus}
            />
          </div>
        )}

        {/* View 2: Plain-Language Statistics with Actual Meaning */}
        {viewMode === "stats" && (
          <LaypersonStats replay={rp} policyName={rp.policy ?? "iql"} overview={ov} />
        )}

        {/* View 3: Time-Distance Diagram (Marey Chart) */}
        {(viewMode === "chart" || viewMode === "combined") && (
          <div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 6,
                paddingTop: viewMode === "combined" ? 14 : 0,
                borderTop: viewMode === "combined" ? "1px solid var(--border)" : "none",
              }}
            >
              <span className="small muted">
                Technical Marey Chart: Time (left to right) vs. Corridor Distance (top to bottom). Horizontal gold bars indicate freight holding in station loops.
              </span>
              <Legend
                items={[
                  ...Object.entries(CLASS_COLORS).map(([k, c]) => ({ label: k, color: c })),
                  { label: "in loop", color: "var(--c-loop)" },
                ]}
              />
            </div>
            <StringDiagram
              replay={rp}
              stations={stations}
              lengthKm={ov.length_km}
              playhead={t}
              onSeek={setT}
              focus={focus}
            />
          </div>
        )}
      </section>

      {/* Train Schedule & Live Status Grid */}
      <div className="grid2">
        <section className="card">
          <div className="card-head">
            <h2>Train Fleet &amp; Punctuality ({rp.trains?.length ?? 0} Trains)</h2>
            <span className="small muted">Click any train to track on the corridor radar</span>
          </div>
          <div className="scroll tall">
            <table className="table compact clickable">
              <thead>
                <tr>
                  <th>Train</th>
                  <th>Service Type</th>
                  <th className="num">Entered</th>
                  <th className="num">Delay</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {(rp.trains ?? []).map((tr: Json) => {
                  const isSelected = focus === tr.tid;
                  const isPax = tr.kind === "passenger";
                  return (
                    <tr
                      key={tr.tid}
                      className={isSelected ? "focus" : ""}
                      onClick={() => setFocus(isSelected ? null : tr.tid)}
                    >
                      <td>
                        <i className="swatch" style={{ background: CLASS_COLORS[tr.cls] }} />
                        <b>{tr.label}</b>
                      </td>
                      <td>
                        <span style={{ fontSize: 11.5, color: isPax ? "var(--accent)" : "var(--muted)" }}>
                          {isPax ? "Passenger Express" : "Freight Cargo"} ({tr.speed_kmh} km/h)
                        </span>
                      </td>
                      <td className="num">{tr.entered >= 0 ? clock(tr.entered) : "–"}</td>
                      <td className={`num ${tr.delay > 15 ? "warn" : ""}`}>{fmt(tr.delay)} min</td>
                      <td>
                        <span
                          style={{
                            display: "inline-block",
                            padding: "1px 6px",
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 600,
                            background:
                              tr.status === "exited"
                                ? "var(--surface-2)"
                                : tr.status === "loop"
                                ? "#fef3c7"
                                : "#ecfdf5",
                            color:
                              tr.status === "exited"
                                ? "var(--muted)"
                                : tr.status === "loop"
                                ? "#b45309"
                                : "#047857",
                          }}
                        >
                          {tr.status === "loop" ? "in siding" : tr.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <h2>Live Dispatch Decisions (Window around {clock(t)})</h2>
            <span className="small muted">Station precedence actions executed</span>
          </div>
          {recent.length === 0 ? (
            <div style={{ padding: "32px 0", textAlign: "center" }} className="muted">
              🟢 Corridor running smoothly. No freight approach conflicts around {clock(t)}.
            </div>
          ) : (
            <div className="scroll tall">
              <table className="table compact">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Station</th>
                    <th>Freight</th>
                    <th>Action</th>
                    <th className="num">Pax ETA</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map((d: Json, i: number) => (
                    <tr key={i}>
                      <td>{clock(d.t)}</td>
                      <td>
                        <b>{stations[d.agent]?.code}</b>
                      </td>
                      <td>{byId.get(d.freight)?.label}</td>
                      <td>
                        <span className={`action a-${ACTION_NAMES[d.action].toLowerCase()}`}>
                          {ACTION_NAMES[d.action]}
                        </span>
                        {!d.valid && <span className="muted"> (siding occupied)</span>}
                        {d.hold_timed_out && <span className="muted"> (hold timeout, released)</span>}
                      </td>
                      <td className="num">{fmt(d.eta)} min</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </>
  );
}
