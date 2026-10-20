import { useState } from "react";
import { digitsFor, fmt, POLICY_COLORS, type Json } from "../api";
import { IntervalPlot } from "../charts/Interval";

export function Results({ ev }: { ev: Json }) {
  const [metric, setMetric] = useState("passenger_delay");
  const meta = ev.metrics.find((m: Json) => m.key === metric) ?? ev.metrics[0];
  const digits = digitsFor(meta.unit);

  return (
    <>
      {/* Dynamic Metric Input Selector */}
      <section className="card">
        <div className="card-head">
          <h2>Policy Performance Comparison</h2>
          <div className="seg">
            {ev.metrics.map((m: Json) => (
              <button
                key={m.key}
                className={m.key === metric ? "on" : ""}
                onClick={() => setMetric(m.key)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Dynamic Interval Plot Output */}
        <IntervalPlot
          unit={meta.unit}
          digits={digits}
          zero={meta.unit !== "reward"}
          rows={ev.order.map((k: string) => ({
            key: k,
            label: ev.labels[k],
            color: POLICY_COLORS[k],
            mean: ev.policies[k][metric].mean,
            lo: ev.policies[k][metric].lo,
            hi: ev.policies[k][metric].hi,
            points: ev.policies[k][metric].per_run,
          }))}
        />
      </section>

      {/* Dynamic Scoreboard Table Output */}
      <section className="card">
        <div className="card-head">
          <h2>Evaluation Summary Matrix</h2>
          <span className="small muted">Evaluated over {ev.protocol.n_runs} held-out runs</span>
        </div>
        <div className="scroll">
          <table className="table compact">
            <thead>
              <tr>
                <th>Policy</th>
                {ev.metrics.map((m: Json) => (
                  <th key={m.key} className="num">
                    {m.label}
                    <br />
                    <span className="muted">{m.unit}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ev.order.map((k: string) => (
                <tr key={k}>
                  <td>
                    <i className="swatch" style={{ background: POLICY_COLORS[k] }} />
                    <b>{ev.labels[k]}</b>
                  </td>
                  {ev.metrics.map((m: Json) => {
                    const s = ev.policies[k][m.key];
                    const dg = digitsFor(m.unit);
                    return (
                      <td key={m.key} className="num">
                        <b>{fmt(s.mean, dg)}</b>
                        <div className="ci">[{fmt(s.lo, dg)}, {fmt(s.hi, dg)}]</div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
