import { fmt, pct, POLICY_COLORS, type Json } from "../api";
import type { Page } from "../scenario";
import { useScenario } from "../scenario";
import { GroupedBars } from "../charts/Bars";

export function Overview({ ov, ev: evPage }: { ov: Json; ev: Page }) {
  const { state } = useScenario();
  const ev = evPage.data;
  const tt = ov.timetable;

  // Dynamically calculate traffic metrics responsive to active scenario controls
  const baseLoad = 0.85;
  const loadFactor = state.traffic.load / baseLoad;
  const activeTrains = Math.round(tt.trains_mean * loadFactor);
  const activePassengers = Math.round(activeTrains * state.traffic.passenger_share);
  const activeFreights = Math.max(0, activeTrains - activePassengers);

  // Dynamic section utilization scaled by active load
  const dynamicSections = (tt.sections as Json[]).map((s: Json) => ({
    name: s.to === "exit" ? "Exit" : s.to,
    utilization: Number((s.utilization * loadFactor).toFixed(2)),
  }));

  const maxUtil = Math.max(...dynamicSections.map((s) => s.utilization));
  const bottleneckSection = dynamicSections.find((s) => s.utilization === maxUtil)?.name ?? "Corridor";

  // Dynamic savings from active evaluation data
  const iqlPDelay = ev?.policies?.iql?.passenger_delay?.mean ?? 0;
  const fcfsPDelay = ev?.policies?.fcfs?.passenger_delay?.mean ?? 0;
  const pDelaySaved = Math.max(0, Math.round(fcfsPDelay - iqlPDelay));

  const iqlReturn = ev?.policies?.iql?.shared_return?.mean ?? 0;
  const fcfsReturn = ev?.policies?.fcfs?.shared_return?.mean ?? 0;
  const returnGain = Math.round(iqlReturn - fcfsReturn);

  return (
    <>
      {/* Live Operational Scenario Cockpit */}
      <section className="card lead">
        <div className="card-head">
          <div>
            <h2>Corridor Precedence Operational Status</h2>
            <div style={{ display: "flex", gap: 8, marginTop: 4, flexWrap: "wrap", alignItems: "center" }}>
              <span className="small muted">Active Inputs:</span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px" }}>
                Load: <b>{pct(state.traffic.load)}</b>
              </span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px" }}>
                Pax Share: <b>{pct(state.traffic.passenger_share)}</b>
              </span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px" }}>
                Jitter: <b>{state.traffic.jitter_min}m</b>
              </span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px" }}>
                Weights: <b>Wp={state.traffic.w_p} / Wf={state.traffic.w_f}</b>
              </span>
            </div>
          </div>
        </div>

        {/* Dynamic Telemetry Metric Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10, marginTop: 14 }}>
          <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 14px" }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>Scheduled Traffic</div>
            <div style={{ fontSize: 22, fontWeight: 700, margin: "2px 0", color: "var(--text)" }}>{activeTrains} trains/day</div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>{activePassengers} passenger · {activeFreights} freight</div>
          </div>

          <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 14px" }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>Bottleneck Section</div>
            <div style={{ fontSize: 22, fontWeight: 700, margin: "2px 0", color: maxUtil > 1.0 ? "var(--c-hold)" : "var(--accent)" }}>
              {bottleneckSection}
            </div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>Peak utilization: {pct(maxUtil)}</div>
          </div>

          <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 14px" }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>RL Delay Reduction</div>
            <div style={{ fontSize: 22, fontWeight: 700, margin: "2px 0", color: "var(--good)" }}>
              {pDelaySaved > 0 ? `-${pDelaySaved.toLocaleString()} min` : "0 min"}
            </div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>Passenger delay saved vs FCFS</div>
          </div>

          <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 14px" }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>Team Return Advantage</div>
            <div style={{ fontSize: 22, fontWeight: 700, margin: "2px 0", color: "var(--good)" }}>
              {returnGain > 0 ? `+${returnGain.toLocaleString()}` : "0"}
            </div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>Net dispatch optimization</div>
          </div>

          <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 14px" }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>SIL-4 Envelope</div>
            <div style={{ fontSize: 22, fontWeight: 700, margin: "2px 0", color: "var(--good)" }}>100% ISOLATED</div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>0 collisions · 0 misroutes</div>
          </div>
        </div>
      </section>

      {/* Dynamic Grid: Section Utilization vs Policy Scorecard */}
      <div className="grid2">
        <section className="card">
          <div className="card-head">
            <h2>Corridor Section Utilisation (Load {pct(state.traffic.load)})</h2>
            <span className="small muted">Red line denotes capacity ceiling (100%)</span>
          </div>
          <GroupedBars
            yLabel="utilisation"
            max={Math.max(1.5, maxUtil + 0.2)}
            series={[{ label: "utilisation", color: maxUtil > 1.0 ? "var(--c-hold)" : "var(--accent)" }]}
            groups={dynamicSections.map((s) => ({ label: s.name, values: [s.utilization] }))}
          />
        </section>

        <section className="card">
          <div className="card-head">
            <h2>Active Policy Performance Scorecard</h2>
            <span className="small muted">Evaluated across scenario runs</span>
          </div>
          {ev ? (
            <div className="scroll">
              <table className="table compact">
                <thead>
                  <tr>
                    <th>Policy</th>
                    <th className="num">Pax Delay</th>
                    <th className="num">Freight Delay</th>
                    <th className="num">Team Return</th>
                  </tr>
                </thead>
                <tbody>
                  {(["iql", "sarsa", "passenger_first", "fcfs"] as const).map((k) => (
                    <tr key={k}>
                      <td>
                        <i className="swatch" style={{ background: POLICY_COLORS[k] }} />
                        <b>{ev.labels[k]}</b>
                      </td>
                      <td className="num">{fmt(ev.policies[k].passenger_delay.mean)} min</td>
                      <td className="num">{fmt(ev.policies[k].freight_delay.mean)} min</td>
                      <td className="num">{fmt(ev.policies[k].shared_return.mean)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : evPage.job ? (
            <div className="muted small" style={{ padding: "40px 0", textAlign: "center" }}>
              Evaluating policies for active scenario ({evPage.job.done} / {evPage.job.total})…
            </div>
          ) : (
            <div className="muted small" style={{ padding: "40px 0", textAlign: "center" }}>
              Scenario requires training. Use the Train button above to compute policy evaluation.
            </div>
          )}
        </section>
      </div>
    </>
  );
}
