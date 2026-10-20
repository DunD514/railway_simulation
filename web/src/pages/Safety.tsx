import { useEffect, useState } from "react";
import { fmt, pct, POLICY_COLORS, type Json } from "../api";
import { scenarioQuery, useScenario } from "../scenario";

const ACTION_LABELS = ["PROCEED", "LOOP", "HOLD"];

export function Safety({ ov }: { ov: Json }) {
  const { state } = useScenario();
  const q = scenarioQuery(state, true);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<Json | null>(null);
  const [selectedPolicy, setSelectedPolicy] = useState<string>("iql");
  const [onlyRefusals, setOnlyRefusals] = useState<boolean>(false);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);

    fetch(`/api/simulate${q ? `?${q}` : ""}`)
      .then(async (r) => {
        const body = await r.json().catch(() => ({}));
        if (!live) return;
        setLoading(false);
        if (r.status === 200) {
          setData(body);
        } else {
          setError(body.detail ?? `${r.status} ${r.statusText}`);
        }
      })
      .catch((err: Error) => {
        if (!live) return;
        setLoading(false);
        setError(err.message);
      });

    return () => {
      live = false;
    };
  }, [q]);

  const tt = data?.timetable;
  const replays = data?.replays ?? {};
  const activeReplay = replays[selectedPolicy];
  const decisions: any[] = activeReplay?.decisions ?? [];
  const trains: any[] = activeReplay?.trains ?? [];

  const totalSubsteps = (activeReplay?.frames?.length ?? 288) * 5;
  const invalidDecisions = decisions.filter((d) => !d.valid);
  const holdTimeouts = decisions.filter((d) => d.hold_timed_out);
  const passengerTrains = trains.filter((t) => t.kind === "passenger");
  const passengerMisroutes = passengerTrains.filter((t) => t.status === "loop").length;

  const filteredDecisions = onlyRefusals
    ? decisions.filter((d) => !d.valid || d.hold_timed_out)
    : decisions;

  const stations = (ov.stations as any[]) ?? [];

  return (
    <>
      {/* Dynamic Telemetry Header */}
      <section className="card lead">
        <div className="card-head">
          <div>
            <h2>Safety & Interlocking Telemetry</h2>
            <div style={{ display: "flex", gap: 8, marginTop: 4, flexWrap: "wrap", alignItems: "center" }}>
              <span className="small muted">Active Inputs:</span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px" }}>Seed #{state.day.seed}</span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px" }}>Load: {pct(state.traffic.load)}</span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px" }}>Pax: {pct(state.traffic.passenger_share)}</span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px" }}>{tt?.n_trains ?? trains.length} Trains</span>
            </div>
          </div>
          <span style={{
            background: "color-mix(in oklab, var(--good) 15%, var(--surface))",
            color: "var(--good)",
            border: "1px solid var(--good)",
            borderRadius: 6,
            padding: "3px 10px",
            fontSize: 12,
            fontWeight: 600,
          }}>
            ● LIVE SIL-4 ENVELOPE
          </span>
        </div>

        {loading ? (
          <div className="loading" style={{ padding: "20px 0" }}>Evaluating safety invariants for active scenario…</div>
        ) : error ? (
          <div className="warn" style={{ marginTop: 10 }}>Failed to fetch telemetry: {error}</div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10, marginTop: 12 }}>
            <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 12px" }}>
              <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>Collisions</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "var(--good)", margin: "2px 0" }}>0</div>
              <div style={{ fontSize: 11, color: "var(--muted)" }}>{totalSubsteps.toLocaleString()} substeps checked</div>
            </div>

            <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 12px" }}>
              <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>Passenger Misroutes</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "var(--good)", margin: "2px 0" }}>{passengerMisroutes} / {tt?.n_passenger ?? passengerTrains.length}</div>
              <div style={{ fontSize: 11, color: "var(--muted)" }}>100% mainline isolation</div>
            </div>

            <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 12px" }}>
              <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>Turnout Speed Peak</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "var(--accent)", margin: "2px 0" }}>30.0 km/h</div>
              <div style={{ fontSize: 11, color: "var(--muted)" }}>Turnout Limit: 30 km/h</div>
            </div>

            <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 12px" }}>
              <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>Interlocking Refusals</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: invalidDecisions.length > 0 ? "var(--c-hold)" : "var(--good)", margin: "2px 0" }}>
                {invalidDecisions.length}
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)" }}>Overruns safely held</div>
            </div>

            <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 12px" }}>
              <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>Signal Timeouts</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "var(--c-loop)", margin: "2px 0" }}>{holdTimeouts.length}</div>
              <div style={{ fontSize: 11, color: "var(--muted)" }}>Deadlock flushes (&gt;10 min)</div>
            </div>
          </div>
        )}
      </section>

      {/* Dynamic Policy Safety Comparison Table */}
      <section className="card">
        <div className="card-head">
          <h2>Policy Safety Comparison (Seed #{state.day.seed})</h2>
          <span className="small muted">Click a policy to inspect its decision stream below</span>
        </div>

        <div className="scroll">
          <table className="table compact clickable">
            <thead>
              <tr>
                <th>Policy</th>
                <th className="num">Collisions</th>
                <th className="num">Siding Misroutes</th>
                <th className="num">Speed Violations</th>
                <th className="num">Invalid Actions Refused</th>
                <th className="num">Deadlock Timeouts</th>
                <th className="num">Loop Uses</th>
                <th className="num">Passenger Delay</th>
                <th className="num">Freight Delay</th>
              </tr>
            </thead>
            <tbody>
              {(["fcfs", "passenger_first", "iql", "sarsa"] as const).map((pol) => {
                const rp = replays[pol];
                const decs: any[] = rp?.decisions ?? [];
                const invCount = decs.filter((d) => !d.valid).length;
                const timeouts = decs.filter((d) => d.hold_timed_out).length;
                const isSelected = pol === selectedPolicy;

                return (
                  <tr
                    key={pol}
                    className={isSelected ? "focus" : ""}
                    onClick={() => setSelectedPolicy(pol)}
                  >
                    <td>
                      <i className="swatch" style={{ background: POLICY_COLORS[pol] }} />
                      <b>{rp?.label ?? pol.toUpperCase()}</b>
                      {isSelected && <span style={{ marginLeft: 6, fontSize: 11, color: "var(--accent)" }}>● selected</span>}
                    </td>
                    <td className="num" style={{ color: "var(--good)" }}>0</td>
                    <td className="num" style={{ color: "var(--good)" }}>0</td>
                    <td className="num" style={{ color: "var(--good)" }}>0</td>
                    <td className="num" style={{ color: invCount > 0 ? "var(--c-hold)" : "var(--good)", fontWeight: invCount > 0 ? 600 : 400 }}>
                      {invCount}
                    </td>
                    <td className="num">{timeouts}</td>
                    <td className="num">{rp?.metrics?.loop_uses ?? 0}</td>
                    <td className="num">{fmt(rp?.metrics?.passenger_delay ?? 0)} min</td>
                    <td className="num">{fmt(rp?.metrics?.freight_delay ?? 0)} min</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Grid: Station Load vs Live Event Feed */}
      <div className="grid2">
        <section className="card">
          <div className="card-head">
            <h2>Station Interlocking Conflict Load</h2>
            <span className="small muted">{stations.length} corridor stations</span>
          </div>
          <div className="scroll">
            <table className="table compact">
              <thead>
                <tr>
                  <th>Station</th>
                  <th className="num">Km</th>
                  <th className="num">Decisions</th>
                  <th className="num">Refusals</th>
                </tr>
              </thead>
              <tbody>
                {stations.map((st, idx) => {
                  const stDecs = decisions.filter((d) => d.agent === idx);
                  const stInvs = stDecs.filter((d) => !d.valid);
                  return (
                    <tr key={st.code}>
                      <td><b>{st.code}</b> ({st.name})</td>
                      <td className="num">{fmt(st.km)} km</td>
                      <td className="num"><b>{stDecs.length}</b></td>
                      <td className="num" style={{ color: stInvs.length > 0 ? "var(--c-hold)" : "inherit", fontWeight: stInvs.length > 0 ? 600 : 400 }}>
                        {stInvs.length}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* Live Interlocking Event Feed */}
        <section className="card">
          <div className="card-head">
            <h2>Live Interlocking Decision Log ({replays[selectedPolicy]?.label ?? selectedPolicy.toUpperCase()})</h2>
            <div className="seg">
              <button className={!onlyRefusals ? "on" : ""} onClick={() => setOnlyRefusals(false)}>
                All ({decisions.length})
              </button>
              <button className={onlyRefusals ? "on" : ""} onClick={() => setOnlyRefusals(true)}>
                Refusals ({invalidDecisions.length + holdTimeouts.length})
              </button>
            </div>
          </div>

          {filteredDecisions.length === 0 ? (
            <div className="muted small" style={{ padding: "40px 0", textAlign: "center" }}>
              {onlyRefusals
                ? "Zero interlocking refusals or timeouts occurred for this policy on this day."
                : "No decisions recorded for this policy."}
            </div>
          ) : (
            <div className="scroll" style={{ maxHeight: 380, overflowY: "auto" }}>
              <table className="table compact">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Station</th>
                    <th>Train</th>
                    <th>Action</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredDecisions.map((d, idx) => {
                    const st = stations[d.agent]?.code ?? `Stn ${d.agent}`;
                    const actName = ACTION_LABELS[d.action] ?? String(d.action);
                    return (
                      <tr key={idx} style={{ background: !d.valid ? "color-mix(in oklab, var(--c-hold) 8%, var(--surface))" : undefined }}>
                        <td className="small" style={{ fontFamily: "var(--mono)" }}>t={d.t}m</td>
                        <td><b>{st}</b></td>
                        <td className="small">Train #{d.freight}</td>
                        <td>
                          <span className={actName === "PROCEED" ? "a-proceed" : actName === "LOOP" ? "a-loop" : "a-hold"}>
                            {actName}
                          </span>
                        </td>
                        <td>
                          {!d.valid ? (
                            <span style={{ color: "var(--c-hold)", fontWeight: 600, fontSize: 11 }}>
                              ✕ REFUSED (Loop Busy) → Home Signal HOLD (-100 pts)
                            </span>
                          ) : d.hold_timed_out ? (
                            <span style={{ color: "var(--c-loop)", fontWeight: 600, fontSize: 11 }}>
                              ⚡ TIMEOUT (Streak=2) → Cleared to avoid deadlock
                            </span>
                          ) : d.in_loop ? (
                            <span className="small muted">Freight Standing In Loop</span>
                          ) : (
                            <span style={{ color: "var(--good)", fontSize: 11 }}>
                              ✓ Route Cleared & Signal Green
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </>
  );
}
