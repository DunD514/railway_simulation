import { useMemo, useState } from "react";
import { clock, type Json } from "../api";
import { linear, Responsive, Tooltip, type Tip } from "./common";

export const CLASS_COLORS: Record<string, string> = {
  VB: "var(--c-vb)",
  RAJ: "var(--c-raj)",
  EXP: "var(--c-exp)",
  FRT: "var(--c-frt)",
};

const IN_LOOP = 3;

interface Pt { t: number; pos: number; status: number }

/** Per train: (time, position, status) samples from epoch frames plus exact station events. */
export function trajectories(replay: Json): Map<number, Pt[]> {
  const out = new Map<number, Pt[]>();
  const push = (tid: number, p: Pt) => {
    let arr = out.get(tid);
    if (!arr) out.set(tid, (arr = []));
    arr.push(p);
  };
  for (const f of replay.frames) for (const [tid, pos, status] of f.trains) push(tid, { t: f.t, pos, status });
  for (const [tid, t, pos, kind] of replay.events) {
    push(tid, { t, pos, status: kind === "loop_in" ? IN_LOOP : kind === "exit" ? 4 : 1 });
  }
  for (const arr of out.values()) arr.sort((a, b) => a.t - b.t || a.pos - b.pos);
  return out;
}

/**
 * Time-distance ("string") diagram: time runs left to right, distance top to bottom.
 * Steeper lines are faster trains; flat segments are trains standing (dwell, hold, loop).
 */
export function StringDiagram({
  replay, stations, lengthKm, playhead, onSeek, focus,
}: {
  replay: Json;
  stations: { code: string; km: number }[];
  lengthKm: number;
  playhead: number;
  onSeek: (t: number) => void;
  focus: number | null;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const traj = useMemo(() => trajectories(replay), [replay]);
  const tMax = useMemo(() => Math.max(...replay.trains.map((t: Json) => (t.exit > 0 ? t.exit : 0)), replay.frames.at(-1).t), [replay]);
  const byId = useMemo(() => new Map<number, Json>(replay.trains.map((t: Json) => [t.tid, t])), [replay]);
  const height = 460;
  const m = { l: 52, r: 12, t: 12, b: 28 };
  return (
    <Responsive height={height}>
      {(w) => {
        const x = linear(0, tMax, m.l, w - m.r);
        const y = linear(0, lengthKm, m.t, height - m.b);
        const hours = Array.from({ length: Math.floor(tMax / 60) + 1 }, (_, h) => h * 60);
        const every = w < 600 ? 4 : 2;
        return (
          <>
            <svg width={w} height={height} className="string"
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                onSeek(Math.max(0, Math.min(tMax, x.invert(e.clientX - r.left))));
              }}
              onMouseLeave={() => setTip(null)}>
              {hours.map((h) => (
                <g key={h}>
                  <line className="grid" x1={x(h)} x2={x(h)} y1={m.t} y2={height - m.b} />
                  {(h / 60) % every === 0 && <text className="tick" x={x(h)} y={height - m.b + 16} textAnchor="middle">{clock(h)}</text>}
                </g>
              ))}
              {stations.map((s) => (
                <g key={s.code}>
                  <line className="station-line" x1={m.l} x2={w - m.r} y1={y(s.km)} y2={y(s.km)} />
                  <text className="tick strong" x={m.l - 6} y={y(s.km)} dy="0.32em" textAnchor="end">{s.code}</text>
                </g>
              ))}
              {[...traj.entries()].map(([tid, pts]) => {
                const tr = byId.get(tid);
                const color = CLASS_COLORS[tr.cls] ?? "var(--muted)";
                const dim = focus !== null && focus !== tid;
                const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.pos).toFixed(1)}`).join("");
                const loops: string[] = [];
                for (let i = 1; i < pts.length; i++) {
                  if (pts[i - 1].status === IN_LOOP && pts[i].status === IN_LOOP) {
                    loops.push(`M${x(pts[i - 1].t).toFixed(1)},${y(pts[i - 1].pos).toFixed(1)}L${x(pts[i].t).toFixed(1)},${y(pts[i].pos).toFixed(1)}`);
                  }
                }
                return (
                  <g key={tid} opacity={dim ? 0.15 : 1}
                    onMouseMove={(e) => {
                      const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                      setTip({
                        x: e.clientX - r.left, y: e.clientY - r.top,
                        body: <><b>{tr.label}</b><div className="muted">{tr.kind}, {tr.speed_kmh} km/h</div>
                          <div>delay {tr.delay.toLocaleString()} min · {tr.status}</div></>,
                      });
                    }}>
                    <path d={d} fill="none" stroke="transparent" strokeWidth={8} />
                    <path d={d} fill="none" stroke={color} strokeWidth={tr.kind === "freight" ? 1.6 : 1.4} />
                    {loops.length > 0 && <path d={loops.join("")} fill="none" stroke="var(--c-loop)" strokeWidth={5} strokeLinecap="round" opacity={0.85} />}
                  </g>
                );
              })}
              <line className="playhead" x1={x(playhead)} x2={x(playhead)} y1={m.t} y2={height - m.b} />
            </svg>
            <Tooltip tip={tip} width={w} />
          </>
        );
      }}
    </Responsive>
  );
}

/** Snapshot of the corridor at one moment: trains as markers on a straight line. */
export function CorridorStrip({
  replay, stations, lengthKm, t,
}: { replay: Json; stations: { code: string; km: number }[]; lengthKm: number; t: number }) {
  const frame = useMemo(() => {
    const fs = replay.frames;
    let lo = 0;
    let hi = fs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (fs[mid].t <= t) lo = mid;
      else hi = mid - 1;
    }
    return fs[lo];
  }, [replay, t]);
  const byId = useMemo(() => new Map<number, Json>(replay.trains.map((tr: Json) => [tr.tid, tr])), [replay]);
  const height = 92;
  return (
    <Responsive height={height}>
      {(w) => {
        const m = { l: 16, r: 16 };
        const x = linear(0, lengthKm, m.l, w - m.r);
        const mid = 46;
        return (
          <svg width={w} height={height}>
            <line x1={x(0)} x2={x(lengthKm)} y1={mid} y2={mid} className="track" />
            {stations.map((s) => (
              <g key={s.code}>
                <rect x={x(s.km) - 9} y={mid + 6} width={18} height={4} rx={2} className="loop-track" />
                <line x1={x(s.km)} x2={x(s.km)} y1={mid - 8} y2={mid + 8} className="station-tick" />
                <text className="tick strong" x={x(s.km)} y={height - 6} textAnchor="middle">{s.code}</text>
              </g>
            ))}
            {(() => {
              // label a running train only if it is not crowded by the previous labelled one
              let lastX = -Infinity;
              const labelled = new Set<number>();
              for (const [tid, pos, status] of [...frame.trains].sort((a: number[], b: number[]) => a[1] - b[1])) {
                if (status !== IN_LOOP && x(pos) - lastX > 44) {
                  labelled.add(tid);
                  lastX = x(pos);
                }
              }
              return frame.trains.map(([tid, pos, status, , accrued]: number[]) => {
              const tr = byId.get(tid);
              const looped = status === IN_LOOP;
              const cy = looped ? mid + 8 : mid;
              return (
                <g key={tid} transform={`translate(${x(pos)},${cy})`}>
                  <title>{`${tr.label} · ${pos.toFixed(0)} km · delay ${accrued} min${looped ? " · in loop" : ""}`}</title>
                  {tr.kind === "freight"
                    ? <rect x={-6} y={-4} width={12} height={8} rx={1.5} fill={CLASS_COLORS.FRT} stroke="var(--surface)" strokeWidth={1.5} />
                    : <circle r={5} fill={CLASS_COLORS[tr.cls]} stroke="var(--surface)" strokeWidth={1.5} />}
                  {labelled.has(tid) && <text y={-10} textAnchor="middle" className="train-label">{tr.label}</text>}
                </g>
              );
              });
            })()}
          </svg>
        );
      }}
    </Responsive>
  );
}
