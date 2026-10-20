import { useEffect, useState } from "react";
import { digitsFor, fmt, pct, POLICY_COLORS, type Json } from "../api";
import { scenarioQuery, useScenario, useStatus } from "../scenario";
import { ReplayView } from "./ReplayView";

const METRICS = [
  { key: "passenger_delay", label: "Passenger delay", unit: "min", better: "low", desc: "Total traveler delay across all passenger trains (lower is better)" },
  { key: "freight_delay", label: "Freight delay", unit: "min", better: "low", desc: "Total siding pause time absorbed by goods trains (lower is better)" },
  { key: "trains_cleared", label: "Cleared in 24 h", unit: "trains", better: "high", desc: "How many scheduled trains safely traversed the entire 870 km corridor" },
  { key: "loop_uses", label: "Loop uses", unit: "count", better: "", desc: "Number of smart overtakes executed using station siding tracks" },
  { key: "invalid_actions", label: "Invalid actions", unit: "count", better: "low", desc: "Decisions blocked by computer safety interlocking (0 is ideal)" },
  { key: "shared_return", label: "Team reward", unit: "reward", better: "high", desc: "Composite score weighting passenger delay 10x higher than freight" },
];

const sims = new Map<string, Json>();

export function Simulate({ ov }: { ov: Json }) {
  const { state, setDay, version } = useScenario();
  const training = (useStatus().status?.jobs ?? []).some((j: Json) => j.kind === "train");
  const [draft, setDraft] = useState(state.day);
  const [result, setResult] = useState<Json | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [needsTraining, setNeedsTraining] = useState(false);
  const [busy, setBusy] = useState(false);
  const [policy, setPolicy] = useState("iql");
  const [summaryMode, setSummaryMode] = useState<"cards" | "table">("cards");

  const q = scenarioQuery(state, true);
  useEffect(() => setDraft(state.day), [state.day]);

  useEffect(() => {
    let live = true;
    const hit = sims.get(q);
    if (hit) {
      setResult(hit);
      setError(null);
      setNeedsTraining(false);
      return;
    }
    setBusy(true);
    setError(null);
    fetch(`/api/simulate${q ? `?${q}` : ""}`).then(async (r) => {
      const body = await r.json().catch(() => ({}));
      if (!live) return;
      setBusy(false);
      const needs = r.status === 200 && body.status === "needs_training";
      setNeedsTraining(needs);
      if (needs) return;
      if (r.status === 200) {
        sims.set(q, body);
        setResult(body);
      } else {
        setError(body.detail ?? `${r.status} ${r.statusText}`);
      }
    }, (e: Error) => { if (live) { setBusy(false); setError(e.message); } });
    return () => { live = false; };
  }, [q, version]);

  const dirty = draft.seed !== state.day.seed || draft.model_run !== state.day.model_run;
  const tt = result?.timetable;
  const nRuns = result?.n_model_runs ?? 10;

  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h2>Simulate &amp; Dispatch Day</h2>
            <span className="small muted">Run simulated rail operations live on the 870 km corridor</span>
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <span className="small muted">Preset Scenarios:</span>
            {[
              { name: "Day #1 (Balanced)", seed: 900000000 },
              { name: "Day #2 (Heavy Rush)", seed: 900000001 },
              { name: "Day #3 (Freight Surge)", seed: 900000002 },
            ].map((p) => (
              <button
                key={p.seed}
                type="button"
                className={`btn ${draft.seed === p.seed ? "on" : ""}`}
                style={{ padding: "2px 8px", fontSize: "12px", borderRadius: "6px" }}
                onClick={() => {
                  const next = { ...draft, seed: p.seed };
                  setDraft(next);
                  setDay(next);
                }}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        <form className="day-form" onSubmit={(e) => { e.preventDefault(); setDay(draft); }}>
          <label className="field">
            <span>Timetable Seed</span>
            <span className="seed-row">
              <input type="number" min={0} max={2147483647} step={1} value={draft.seed}
                onChange={(e) => setDraft({ ...draft, seed: Math.max(0, Math.floor(+e.target.value || 0)) })} />
              <button type="button" className="btn" onClick={() => setDraft({ ...draft, seed: Math.floor(Math.random() * 1e6) })}>Random Seed</button>
            </span>
          </label>
          <label className="field">
            <span>Model Weights Run: <b>Run #{draft.model_run}</b> (of {nRuns})</span>
            <input type="range" min={0} max={Math.max(0, nRuns - 1)} step={1} value={Math.min(draft.model_run, nRuns - 1)}
              onChange={(e) => setDraft({ ...draft, model_run: +e.target.value })} />
          </label>
          <div className="form-actions">
            <button type="submit" className="btn primary" disabled={busy || !dirty}>{busy ? "Simulating…" : "Run this day"}</button>
          </div>
        </form>
      </section>

      {needsTraining && (
        <section className="card notice">
          <h2>{training ? "Training agents on this scenario" : "No agents trained on this scenario yet"}</h2>
          <p className="muted">{training ? "This day is simulated as soon as training finishes (progress above)."
            : "Train them with the Train button above, or switch the agents to \"trained on default traffic\"."}</p>
        </section>
      )}
      {error && <section className="card error"><h2>Could not simulate</h2><p>{error}</p></section>}

      {result && !needsTraining && (
        <section className={`card ${busy ? "stale" : ""}`}>
          <div className="card-head">
            <div>
              <h2>Dispatch Strategies Comparison</h2>
              <span className="small muted">Compare AI against conventional railway dispatch methods</span>
            </div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
              <div className="seg">
                <button className={summaryMode === "cards" ? "on" : ""} onClick={() => setSummaryMode("cards")}>
                  Plain-English Cards
                </button>
                <button className={summaryMode === "table" ? "on" : ""} onClick={() => setSummaryMode("table")}>
                  Detailed Metrics Table
                </button>
              </div>

              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "2px 8px" }}>
                <b>{tt.n_trains}</b> Trains ({tt.n_passenger} Pax / {tt.n_freight} Frt)
              </span>
              <span style={{ fontSize: 12, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "2px 8px" }}>
                Line Load: <b>{pct(tt.utilization)}</b>
              </span>
            </div>
          </div>

          {/* Meaningful Cards View */}
          {summaryMode === "cards" ? (
            <div className="strategy-cards-grid">
              {result.policies.map((p: string) => {
                const rp = result.replays[p];
                const mt = rp.metrics;
                const isSelected = p === policy;
                const paxDelay = mt.passenger_delay ?? 0;
                const frtDelay = mt.freight_delay ?? 0;
                const paxAvg = paxDelay / Math.max(1, tt.n_passenger);
                const loopUses = mt.loop_uses ?? 0;

                return (
                  <div
                    key={p}
                    className={`strategy-card ${isSelected ? "selected" : ""}`}
                    onClick={() => setPolicy(p)}
                  >
                    <div className="strategy-card-head">
                      <div className="strategy-title-row">
                        <i className="swatch" style={{ background: POLICY_COLORS[p] }} />
                        <h3>{result.labels[p]}</h3>
                      </div>
                      {isSelected && <span className="active-pill">● REPLAY ACTIVE</span>}
                    </div>

                    <p className="strategy-desc">
                      {p === "iql"
                        ? "Independent Q-Learning: AI balances express passenger speed with freight flow."
                        : p === "sarsa"
                        ? "SARSA On-Policy AI: Learns conservative, deadlock-free corridor precedence."
                        : p === "passenger_first"
                        ? "Passenger First: Always holds freight in sidings when any passenger train approaches."
                        : "First-Come First-Served: No precedence. Fast trains get trapped behind slow freight."}
                    </p>

                    <div className="strategy-kpis">
                      <div className="strategy-kpi-item">
                        <span className="kpi-micro-label">Avg Pax Delay</span>
                        <span className={`kpi-micro-val ${paxAvg <= 15 ? "good" : "warn"}`}>
                          {paxAvg.toFixed(1)} min
                        </span>
                      </div>
                      <div className="strategy-kpi-item">
                        <span className="kpi-micro-label">Freight Pause</span>
                        <span className="kpi-micro-val">{(frtDelay / Math.max(1, tt.n_freight)).toFixed(0)} min</span>
                      </div>
                      <div className="strategy-kpi-item">
                        <span className="kpi-micro-label">Overtakes</span>
                        <span className="kpi-micro-val">{loopUses}</span>
                      </div>
                    </div>

                    <div className="strategy-card-footer">
                      <button type="button" className={`btn small ${isSelected ? "primary" : ""}`}>
                        {isSelected ? "Currently Viewing" : "Load this Dispatch"}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            /* Table View with Explanatory Tooltips */
            <div className="scroll">
              <table className="table clickable">
                <thead>
                  <tr>
                    <th>Policy</th>
                    {METRICS.map((m) => (
                      <th key={m.key} className="num" title={m.desc}>
                        {m.label}
                        <br />
                        <span className="muted">{m.unit}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.policies.map((p: string) => {
                    const mt = result.replays[p].metrics;
                    const isSelected = p === policy;
                    return (
                      <tr key={p} className={isSelected ? "focus" : ""} onClick={() => setPolicy(p)}>
                        <td>
                          <i className="swatch" style={{ background: POLICY_COLORS[p] }} />
                          <b>{result.labels[p]}</b>
                          {isSelected && <span style={{ marginLeft: 6, fontSize: 11, color: "var(--accent)" }}>● replay active</span>}
                        </td>
                        {METRICS.map((m) => {
                          const vals = result.policies.map((x: string) => result.replays[x].metrics[m.key]);
                          const best = m.better === "low" ? Math.min(...vals) : m.better === "high" ? Math.max(...vals) : null;
                          return (
                            <td key={m.key} className={`num ${best !== null && mt[m.key] === best ? "best" : ""}`}>
                              {m.key === "trains_cleared" ? `${mt.trains_cleared} / ${mt.n_trains}` : fmt(mt[m.key], digitsFor(m.unit) && m.unit !== "count" ? 1 : 0)}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {result && !needsTraining && <ReplayView key={q} rp={result.replays[policy]} ov={ov} />}
    </>
  );
}
