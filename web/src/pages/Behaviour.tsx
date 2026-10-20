import { ACTION_COLORS, ACTION_NAMES, POLICY_COLORS, type Json } from "../api";
import { StackedShares } from "../charts/Bars";
import { Legend } from "../charts/common";

export function Behaviour({ beh }: { beh: Json }) {
  const pols = ["iql", "sarsa", "passenger_first"];
  const parts = ACTION_NAMES.map((n, i) => ({ label: n, color: ACTION_COLORS[i] }));

  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h2>Approach Decisions by Station</h2>
            <span className="small muted">Distribution of PROCEED, LOOP, and HOLD actions per station</span>
          </div>
          <Legend items={parts} />
        </div>
        <div className="grid3">
          {pols.map((p) => (
            <div key={p}>
              <h3><i className="swatch" style={{ background: POLICY_COLORS[p] }} />{beh.labels[p]}</h3>
              <StackedShares
                parts={parts}
                rows={beh.policies[p].per_agent.map((a: Json) => ({ label: beh.stations[a.agent], values: a.approach_counts }))}
              />
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

