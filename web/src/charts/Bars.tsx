import { useState } from "react";
import { compact, extent, linear, Responsive, ticks, Tooltip, type Tip } from "./common";

export interface BarGroup {
  label: string;
  values: (number | null)[];
}

/** Vertical grouped bars (one bar per series inside each category). */
export function GroupedBars({
  groups, series, height = 240, yLabel, digits = 0, max,
}: {
  groups: BarGroup[];
  series: { label: string; color: string }[];
  height?: number;
  yLabel: string;
  digits?: number;
  max?: number;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const m = { l: 52, r: 8, t: 10, b: 30 };
  const vals = groups.flatMap((g) => g.values.filter((v): v is number => v !== null));
  const [, top] = extent(vals, 0.06, true);
  const y1 = max ?? top;
  return (
    <Responsive height={height}>
      {(w) => {
        const y = linear(0, y1, height - m.b, m.t);
        const gw = (w - m.l - m.r) / groups.length;
        const bw = Math.max(3, Math.min(28, (gw - 8) / series.length - 2));
        return (
          <>
            <svg width={w} height={height} onMouseLeave={() => setTip(null)}>
              {ticks(0, y1, 4).map((t) => (
                <g key={t}>
                  <line className="grid" x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} />
                  <text className="tick" x={m.l - 6} y={y(t)} dy="0.32em" textAnchor="end">{max === 1 ? `${Math.round(t * 100)}%` : compact(t)}</text>
                </g>
              ))}
              <text className="axis-label" transform={`translate(12,${m.t}) rotate(90)`}>{yLabel}</text>
              {groups.map((g, gi) => {
                const cx = m.l + gw * gi + gw / 2;
                const x0 = cx - (series.length * (bw + 2)) / 2;
                return (
                  <g key={g.label}>
                    <text className="tick" x={cx} y={height - m.b + 16} textAnchor="middle">{g.label}</text>
                    {g.values.map((v, si) => v !== null && (
                      <rect key={si} x={x0 + si * (bw + 2)} y={y(Math.max(0, v))} width={bw}
                        height={Math.max(0.5, y(0) - y(Math.max(0, v)))} rx={2} fill={series[si].color}
                        onMouseEnter={() => setTip({
                          x: x0 + si * (bw + 2) + bw / 2, y: y(v),
                          body: <><b>{g.label}</b><div className="tip-row"><i style={{ background: series[si].color }} />{series[si].label}
                            <span>{max === 1 ? `${(v * 100).toFixed(1)}%` : v.toLocaleString("en-US", { maximumFractionDigits: digits })}</span></div></>,
                        })} />
                    ))}
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

/** One 100%-stacked horizontal bar per row (action shares). */
export function StackedShares({
  rows, parts,
}: { rows: { label: string; values: number[] }[]; parts: { label: string; color: string }[] }) {
  return (
    <div className="stacked">
      {rows.map((r) => {
        const total = r.values.reduce((a, b) => a + b, 0) || 1;
        return (
          <div className="stacked-row" key={r.label}>
            <span className="stacked-label">{r.label}</span>
            <div className="stacked-bar">
              {r.values.map((v, i) => v > 0 && (
                <div key={i} style={{ width: `${(100 * v) / total}%`, background: parts[i].color }}
                  title={`${parts[i].label}: ${v.toLocaleString()} (${((100 * v) / total).toFixed(1)}%)`}>
                  {v / total > 0.1 ? `${Math.round((100 * v) / total)}%` : ""}
                </div>
              ))}
            </div>
            <span className="stacked-n">{total.toLocaleString()}</span>
          </div>
        );
      })}
    </div>
  );
}
