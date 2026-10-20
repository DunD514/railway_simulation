import { useState } from "react";
import { compact, extent, linear, Responsive, ticks, Tooltip, type Tip } from "./common";

export interface IntervalRow {
  key: string;
  label: string;
  color: string;
  mean: number;
  lo: number;
  hi: number;
  points?: number[];
}

/** Horizontal point-and-interval plot (mean with 95% CI, per-run dots behind). */
export function IntervalPlot({
  rows, unit, digits = 0, zero = false, refLine,
}: { rows: IntervalRow[]; unit: string; digits?: number; zero?: boolean; refLine?: number }) {
  const [tip, setTip] = useState<Tip | null>(null);
  const rowH = 30;
  const m = { l: 120, r: 16, t: 6, b: 26 };
  const height = m.t + m.b + rows.length * rowH;
  const vals = rows.flatMap((r) => [r.lo, r.hi, ...(r.points ?? [])]);
  if (refLine !== undefined) vals.push(refLine);
  const [x0, x1] = extent(vals, 0.06, zero);
  return (
    <Responsive height={height}>
      {(w) => {
        const x = linear(x0, x1, m.l, w - m.r);
        return (
          <>
            <svg width={w} height={height} onMouseLeave={() => setTip(null)}>
              {ticks(x0, x1, Math.max(3, Math.floor(w / 110))).map((t) => (
                <g key={t}>
                  <line className="grid" x1={x(t)} x2={x(t)} y1={m.t} y2={height - m.b} />
                  <text className="tick" x={x(t)} y={height - m.b + 15} textAnchor="middle">{compact(t)}</text>
                </g>
              ))}
              {refLine !== undefined && <line className="zero" x1={x(refLine)} x2={x(refLine)} y1={m.t} y2={height - m.b} />}
              {rows.map((r, i) => {
                const cy = m.t + i * rowH + rowH / 2;
                return (
                  <g key={r.key}
                    onMouseEnter={() => setTip({
                      x: x(r.mean), y: cy,
                      body: (
                        <>
                          <b>{r.label}</b>
                          <div>{r.mean.toLocaleString("en-US", { maximumFractionDigits: digits })} {unit}</div>
                          <div className="muted">95% CI [{r.lo.toLocaleString("en-US", { maximumFractionDigits: digits })}, {r.hi.toLocaleString("en-US", { maximumFractionDigits: digits })}]</div>
                          {r.points && <div className="muted">{r.points.length} runs</div>}
                        </>
                      ),
                    })}>
                    <rect x={0} y={cy - rowH / 2} width={w} height={rowH} fill="transparent" />
                    <text className="row-label" x={m.l - 10} y={cy} dy="0.32em" textAnchor="end">{r.label}</text>
                    {r.points?.map((p, j) => (
                      <circle key={j} cx={x(p)} cy={cy + ((j % 5) - 2) * 2.2} r={2.2} fill={r.color} opacity={0.3} />
                    ))}
                    <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke={r.color} strokeWidth={2.5} strokeLinecap="round" />
                    <circle cx={x(r.mean)} cy={cy} r={5} fill={r.color} stroke="var(--surface)" strokeWidth={2} />
                  </g>
                );
              })}
            </svg>
            <Tooltip tip={tip} width={w} />
          </>
        );
      }}
    </Responsive>
  );
}
