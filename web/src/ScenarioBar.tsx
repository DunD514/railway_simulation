import { useEffect, useState, type ReactNode } from "react";
import { fmt, type Json } from "./api";
import {
  fmtDuration, GROUPS, isDefaultTraffic, showValue, TRAFFIC_DEFAULTS, useScenario, useStatus, type Page,
} from "./scenario";

function summary(t: Record<string, number>): string {
  if (isDefaultTraffic(t)) return "Default (config.yaml)";
  return [
    `load ${Math.round(t.load * 100)}%`,
    `${Math.round(t.passenger_share * 100)}% passenger`,
    `mix VB/RAJ/EXP ${t.mix_vb}/${t.mix_raj}/${t.mix_exp}`,
    `${Math.round(t.late_prob * 100)}% late, avg ${t.late_mean_min} min`,
    `jitter ${t.jitter_min} min`,
    `weights ${t.w_p}/${t.w_f}`,
  ].join(" · ");
}

function Progress({ job, queuePosition }: { job: Json; queuePosition?: number }) {
  const frac = job.total ? job.done / job.total : 0;
  return (
    <div className="progress">
      <div className="bar"><div style={{ width: `${Math.round(frac * 100)}%` }} /></div>
      <span className="muted small">
        {job.state === "queued" ? `waiting${queuePosition ? ` (${queuePosition} ahead)` : ""}` :
          `${job.message || job.kind} ${fmt(job.done)} / ${fmt(job.total)}${job.eta_s !== null ? ` · about ${fmtDuration(job.eta_s)} left` : ""}`}
      </span>
    </div>
  );
}

function TrainPanel({ status, refresh }: { status: Json; refresh: () => void }) {
  const { state } = useScenario();
  const lim = status.budget_limits;
  const [b, setB] = useState({ episodes: 100, seeds: 3, ablations: false, sweep: false });
  const [est, setEst] = useState<Json | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const q = new URLSearchParams(Object.entries({ ...state.traffic, ...b }).map(([k, v]) => [k, String(v)])).toString();

  useEffect(() => {
    let live = true;
    fetch(`/api/train/estimate?${q}`).then((r) => r.json()).then((e) => live && setEst(e), () => undefined);
    return () => { live = false; };
  }, [q]);

  const start = async () => {
    setErr(null);
    const r = await fetch(`/api/train?${q}`, { method: "POST" });
    if (r.status >= 400) setErr((await r.json()).detail ?? r.statusText);
    refresh();
  };

  return (
    <div className="train-panel">
      <label className="field">
        <span>Episodes per run <b>{b.episodes}</b></span>
        <input type="range" min={lim.episodes[0]} max={lim.episodes[1]} step={10} value={b.episodes}
          onChange={(e) => setB({ ...b, episodes: +e.target.value })} />
      </label>
      <label className="field">
        <span>Training runs (seeds) <b>{b.seeds}</b></span>
        <input type="range" min={lim.seeds[0]} max={lim.seeds[1]} step={1} value={b.seeds}
          onChange={(e) => setB({ ...b, seeds: +e.target.value })} />
      </label>
      <div className="checks">
        <label><input type="checkbox" checked={b.ablations} onChange={(e) => setB({ ...b, ablations: e.target.checked })} /> include ablations</label>
        <label><input type="checkbox" checked={b.sweep} onChange={(e) => setB({ ...b, sweep: e.target.checked })} /> include hyperparameter sweep</label>
      </div>
      <div className="form-actions">
        <button className="btn primary" onClick={start}>Train on this scenario</button>
        {est && <span className="muted small">{fmt(est.episodes)} episodes on {est.workers} worker{est.workers > 1 ? "s" : ""}, about {fmtDuration(est.seconds)}</span>}
        {err && <span className="warn">{err}</span>}
      </div>
      <p className="muted small">The original experiment used 500 episodes × 10 runs. Fewer episodes train faster but the
        agents learn less; the Learning tab shows how far they got.</p>
    </div>
  );
}

export function ScenarioBar() {
  const { state, setTraffic, setAgents } = useScenario();
  const { status, refresh } = useStatus();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(state.traffic);
  const [showTrain, setShowTrain] = useState(false);
  useEffect(() => setDraft(state.traffic), [state.traffic]);

  const isDefault = isDefaultTraffic(state.traffic);
  const trained = status?.trained;
  const trainJob = (status?.jobs ?? []).find((j: Json) => j.kind === "train");
  const mixZero = draft.mix_vb + draft.mix_raj + draft.mix_exp <= 0;
  const dirty = JSON.stringify(draft) !== JSON.stringify(state.traffic);

  return (
    <section className="scenario-bar">
      <div className="sb-row">
        <span className="sb-label">Scenario</span>
        <span className="sb-value">{summary(state.traffic)}</span>
        <button className="btn small" onClick={() => setEditing(!editing)}>{editing ? "Close" : "Change"}</button>
        {!isDefault && <button className="btn small" onClick={() => setTraffic({ ...TRAFFIC_DEFAULTS })}>Reset</button>}
        {!isDefault && (
          <div style={{ marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" }}>
            <span className="seg">
              <button className={state.agents === "default" ? "on" : ""} onClick={() => setAgents("default")}>default</button>
              <button className={state.agents === "trained" ? "on" : ""} onClick={() => setAgents("trained")}>scenario</button>
            </span>
            {!trainJob && <button className="btn small" onClick={() => setShowTrain(!showTrain)}>{trained ? "Retrain" : "Train"}</button>}
          </div>
        )}
      </div>

      {editing && (
        <form className="sim-form" onSubmit={(e) => { e.preventDefault(); if (!mixZero) { setTraffic(draft); setEditing(false); } }}>
          {GROUPS.map((g) => (
            <fieldset key={g.title}>
              <legend>{g.title}</legend>
              {g.fields.map((f) => (
                <label className="field" key={f.key}>
                  <span>{f.label} <b>{showValue(f, draft[f.key])}</b></span>
                  <input type="range" min={f.min} max={f.max} step={f.step} value={draft[f.key]}
                    onChange={(e) => setDraft({ ...draft, [f.key]: +e.target.value })} />
                  {f.help && <small>{f.help}</small>}
                </label>
              ))}
            </fieldset>
          ))}
          <div className="form-actions">
            <button type="submit" className="btn primary" disabled={mixZero || !dirty}>Apply to all tabs</button>
            <button type="button" className="btn" onClick={() => setDraft({ ...TRAFFIC_DEFAULTS })}>Default values</button>
            {mixZero && <span className="warn">Give at least one passenger class a weight above zero.</span>}
          </div>
        </form>
      )}

      {trainJob && <Progress job={trainJob} />}
      {showTrain && !trainJob && status && !isDefault && <TrainPanel status={status} refresh={() => { setShowTrain(false); refresh(); }} />}
    </section>
  );
}

/** Renders a page once its data is ready; otherwise explains what the server is doing. */
export function PageGate({ pages, children }: { pages: Page[]; children: () => ReactNode }) {
  const { setAgents } = useScenario();
  const { status } = useStatus();
  const trainJob = (status?.jobs ?? []).find((j: Json) => j.kind === "train");
  const err = pages.find((p) => p.error)?.error;
  if (err) return <section className="card error"><h2>Could not load this page</h2><p>{err}</p></section>;
  if (pages.some((p) => p.needsTraining) && trainJob) {
    return (
      <section className="card notice">
        <h2>Training agents on this scenario</h2>
        <p className="muted">IQL and SARSA are learning on days generated with your settings (progress above). When they
          finish they are evaluated on this scenario's held-out days and this page fills in.</p>
      </section>
    );
  }
  if (pages.some((p) => p.needsTraining)) {
    return (
      <section className="card notice">
        <h2>No agents trained on this scenario yet</h2>
        <p>This page shows agents trained on your scenario. Train them with the Train button above, or look at the agents
          trained on the default traffic instead.</p>
        <button className="btn" onClick={() => setAgents("default")}>Use agents trained on default traffic</button>
      </section>
    );
  }
  const pending = pages.find((p) => p.job);
  if (pending) {
    const j = pending.job;
    return (
      <section className="card notice">
        <h2>{j.kind === "train" ? "Training agents on this scenario" : "Computing this scenario"}</h2>
        <p className="muted">{j.kind === "train"
          ? "IQL and SARSA are learning on days generated with your settings. This page fills in when they finish."
          : "Every policy is being evaluated on this scenario's held-out days. All tabs except Learning come from this one evaluation, so they will be ready together."}</p>
        <Progress job={j} queuePosition={pending.queuePosition} />
      </section>
    );
  }
  if (pages.some((p) => !p.data)) return <div className="loading">Loading…</div>;
  return <>{children()}</>;
}
