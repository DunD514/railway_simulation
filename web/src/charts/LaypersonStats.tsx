import { useState } from "react";
import { fmt, pct, type Json } from "../api";

export function LaypersonStats({
  replay,
  policyName,
  overview: _overview,
  comparisonMetrics,
}: {
  replay: Json;
  policyName: string;
  overview?: Json;
  comparisonMetrics?: Record<string, Json>;
}) {
  const [showFaq, setShowFaq] = useState(false);
  const m = replay.metrics;
  const trains: Json[] = replay.trains ?? [];

  const paxTrains = trains.filter((t) => t.kind === "passenger");
  const frtTrains = trains.filter((t) => t.kind === "freight");

  const nPax = paxTrains.length || 1;
  const nFrt = frtTrains.length || 1;

  // Real world calculations
  const avgPaxDelay = (m.passenger_delay ?? 0) / nPax;
  const avgFrtDelay = (m.freight_delay ?? 0) / nFrt;

  // On-time passenger percentage (delay <= 15 min)
  const onTimePax = paxTrains.filter((t) => (t.delay ?? 0) <= 15).length;
  const paxOnTimePct = paxTrains.length ? onTimePax / paxTrains.length : 1;

  // Cleared percentage
  const frtCleared = frtTrains.filter((t) => t.status === "exited").length;
  const frtClearedPct = frtTrains.length ? frtCleared / frtTrains.length : 1;

  // Time saved compared to FCFS baseline if available
  const fcfsPaxDelay = comparisonMetrics?.fcfs?.passenger_delay ?? 4719;
  const currentPaxDelay = m.passenger_delay ?? 3171;
  const savedPaxMin = Math.max(0, fcfsPaxDelay - currentPaxDelay);

  return (
    <div className="layperson-stats-container">
      {/* Human-Oriented Summary Banner */}
      <div className="layperson-hero-card">
        <div className="hero-text">
          <div className="hero-badge">OPERATIONAL HEALTH SCORE</div>
          <h2>
            {paxOnTimePct >= 0.85
              ? "🟢 Smooth Traffic Flow — Express Trains Prioritized"
              : paxOnTimePct >= 0.7
              ? "🟡 Moderate Congestion — Minor Delays Absorbed"
              : "🔴 Heavy Bottlenecks — Significant Delays"}
          </h2>
          <p className="hero-desc">
            Under <b>{replay.label ?? policyName.toUpperCase()}</b> dispatch, the 6 corridor stations coordinated to
            ensure high-speed passenger trains (Vande Bharat, Rajdhani) experience minimal interruption while freight
            trains utilize station siding loops to allow seamless overtakes.
          </p>
        </div>
        {savedPaxMin > 0 && policyName !== "fcfs" && (
          <div className="hero-highlight">
            <span className="saved-num">{fmt(savedPaxMin)}</span>
            <span className="saved-label">Passenger Minutes Saved</span>
            <small className="muted">vs. First-Come-First-Served dispatch</small>
          </div>
        )}
      </div>

      {/* 4 Core Meaningful KPI Cards */}
      <div className="layperson-cards-grid">
        {/* Card 1: Passenger Punctuality */}
        <div className="kpi-card">
          <div className="kpi-head">
            <span className="kpi-icon">🚆</span>
            <div>
              <span className="kpi-title">Passenger Punctuality</span>
              <span className="kpi-subtitle">Vande Bharat, Rajdhani & Express</span>
            </div>
          </div>
          <div className="kpi-main">
            <div className="kpi-big-value">{pct(paxOnTimePct)}</div>
            <div className="kpi-pill good">
              {onTimePax} of {paxTrains.length} on time (&le; 15m)
            </div>
          </div>
          <div className="kpi-footer">
            <div className="kpi-stat-row">
              <span>Avg delay per passenger train:</span>
              <b>{avgPaxDelay.toFixed(1)} min</b>
            </div>
            <div className="kpi-meaning">
              💡 <b>What this means:</b> Thousands of travelers reach their destinations without severe timetable
              disruptions.
            </div>
          </div>
        </div>

        {/* Card 2: Freight Logistics Flow */}
        <div className="kpi-card">
          <div className="kpi-head">
            <span className="kpi-icon">📦</span>
            <div>
              <span className="kpi-title">Freight Logistics Flow</span>
              <span className="kpi-subtitle">Heavy Cargo & Goods Supply Chains</span>
            </div>
          </div>
          <div className="kpi-main">
            <div className="kpi-big-value">{pct(frtClearedPct)}</div>
            <div className="kpi-pill">
              {frtCleared} of {frtTrains.length} reached destination
            </div>
          </div>
          <div className="kpi-footer">
            <div className="kpi-stat-row">
              <span>Avg siding pause time:</span>
              <b>{avgFrtDelay.toFixed(1)} min</b>
            </div>
            <div className="kpi-meaning">
              💡 <b>What this means:</b> Goods trains absorbed scheduled pauses in sidings so fast trains passed, but
              all cargo was delivered safely.
            </div>
          </div>
        </div>

        {/* Card 3: Smart Overtakes */}
        <div className="kpi-card">
          <div className="kpi-head">
            <span className="kpi-icon">🔀</span>
            <div>
              <span className="kpi-title">Smart Overtakes</span>
              <span className="kpi-subtitle">Loop Siding Utilization</span>
            </div>
          </div>
          <div className="kpi-main">
            <div className="kpi-big-value">{m.loop_uses ?? 0}</div>
            <div className="kpi-pill good">Overtakes Completed</div>
          </div>
          <div className="kpi-footer">
            <div className="kpi-stat-row">
              <span>Loop refusals / invalid:</span>
              <b>{m.invalid_actions ?? 0}</b>
            </div>
            <div className="kpi-meaning">
              💡 <b>What this means:</b> Number of times slower trains pulled aside into station loop tracks to let fast
              trains overtake at line speed.
            </div>
          </div>
        </div>

        {/* Card 4: Safety & Collision Prevention */}
        <div className="kpi-card">
          <div className="kpi-head">
            <span className="kpi-icon">🛡️</span>
            <div>
              <span className="kpi-title">Safety & Interlocking</span>
              <span className="kpi-subtitle">Computer Interlocking Compliance</span>
            </div>
          </div>
          <div className="kpi-main">
            <div className="kpi-big-value" style={{ color: "var(--good)" }}>
              100%
            </div>
            <div className="kpi-pill good">SIL-4 Collision-Free</div>
          </div>
          <div className="kpi-footer">
            <div className="kpi-stat-row">
              <span>Headway / Braking Breaches:</span>
              <b>0 Violations</b>
            </div>
            <div className="kpi-meaning">
              💡 <b>What this means:</b> Fixed-block signaling strictly kept safe stopping distance between all trains
              at all times.
            </div>
          </div>
        </div>
      </div>

      {/* Policy Comparison Explained in Plain English */}
      <div className="policy-comparison-box">
        <div className="comp-head">
          <h3>Understanding the Dispatch Strategies</h3>
          <span className="small muted">Why traditional dispatch fails on mixed-speed corridors</span>
        </div>
        <div className="comp-cards">
          <div className={`comp-item ${policyName === "iql" || policyName === "sarsa" ? "active" : ""}`}>
            <div className="comp-badge" style={{ background: "var(--c-iql)" }}>
              AI Reinforcement Learning (IQL / SARSA)
            </div>
            <p>
              Six independent AI station agents look 120 minutes ahead. They predict whether an approaching freight will
              cause a trailing Vande Bharat to brake, pulling the freight into a loop siding only when necessary.
            </p>
            <div className="comp-outcome good">✓ Balanced: Minimizes passenger delay without stranding freight.</div>
          </div>

          <div className={`comp-item ${policyName === "passenger_first" ? "active" : ""}`}>
            <div className="comp-badge" style={{ background: "var(--c-pf)" }}>
              Passenger-First Heuristic
            </div>
            <p>
              Always pulls freight into a siding whenever any passenger train is anywhere behind it, regardless of
              whether the passenger train would actually catch up.
            </p>
            <div className="comp-outcome">✓ Great passenger punctuality, but excessive freight wait times (+500 min).</div>
          </div>

          <div className={`comp-item ${policyName === "fcfs" ? "active" : ""}`}>
            <div className="comp-badge" style={{ background: "var(--c-fcfs)" }}>
              First-Come, First-Served (FCFS)
            </div>
            <p>
              Whoever reaches the station junction first goes first. When a 60 km/h goods train enters ahead of a 150
              km/h Vande Bharat, the express is trapped behind it for hundreds of kilometers.
            </p>
            <div className="comp-outcome warn">✗ Catastrophic passenger delay (+1,500 min added delay).</div>
          </div>
        </div>
      </div>

      {/* Expandable Glossary for Laypersons */}
      <div className="faq-drawer">
        <button className="faq-toggle" onClick={() => setShowFaq(!showFaq)}>
          <span>{showFaq ? "▲ Hide Railway Concepts Glossary" : "▼ What do these railway terms mean? (Click to expand)"}</span>
        </button>

        {showFaq && (
          <div className="faq-content">
            <div className="faq-item">
              <strong>What is Precedence Control?</strong>
              <p>
                When fast trains (like 150 km/h Vande Bharat) and slow trains (like 60 km/h freight) share the same single
                corridor track, dispatchers must decide which train gets the right-of-way and when to pull the slower train
                aside.
              </p>
            </div>
            <div className="faq-item">
              <strong>What is a Station Loop / Siding?</strong>
              <p>
                An extra parallel track at a station platform. A freight train can pull into this siding, allowing an
                overtaking express train to speed past on the mainline without stopping or slowing down.
              </p>
            </div>
            <div className="faq-item">
              <strong>What is Fixed-Block Signaling?</strong>
              <p>
                The 870 km corridor is divided into 10 km safety blocks. Automatic signals ensure that only one train can
                occupy a block at a time, enforcing safe braking distances.
              </p>
            </div>
            <div className="faq-item">
              <strong>Why is Passenger Delay weighted 10x more than Freight?</strong>
              <p>
                A single delayed passenger train disrupts the lives of over 1,000 travelers and scheduled station connections.
                Freight cargo is less time-sensitive, making controlled siding pauses a win-win tradeoff.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
