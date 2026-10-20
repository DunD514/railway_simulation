import { useEffect, useRef, useState, type ReactNode } from "react";

/** Linear scale [d0, d1] -> [r0, r1]. */
export function linear(d0: number, d1: number, r0: number, r1: number) {
  const span = d1 - d0 || 1;
  const f = (x: number) => r0 + ((x - d0) / span) * (r1 - r0);
  f.invert = (y: number) => d0 + ((y - r0) / (r1 - r0)) * span;
  return f;
}

/** About `count` round tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Pad a numeric extent and optionally force zero into it. */
export function extent(values: number[], pad = 0.05, zero = false): [number, number] {
  const finite = values.filter((v) => Number.isFinite(v));
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (zero) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const d = (hi - lo) * pad;
  return [zero && lo === 0 ? 0 : lo - d, zero && hi === 0 ? 0 : hi + d];
}

export function compact(x: number): string {
  const a = Math.abs(x);
  if (a >= 1e4) return `${+(x / 1e3).toFixed(a >= 1e5 ? 0 : 1)}k`;
  if (a >= 100) return x.toFixed(0);
  if (a >= 1) return +x.toFixed(1) + "";
  return +x.toFixed(2) + "";
}

/** Measures its own width so SVG charts can lay out in pixels. */
export function Responsive({ height, children }: { height: number; children: (w: number) => ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={ref} className="chart" style={{ height }}>
      {w > 0 && children(w)}
    </div>
  );
}

export interface Tip {
  x: number;
  y: number;
  body: ReactNode;
}

/** Absolutely positioned tooltip inside a `.chart` container. */
export function Tooltip({ tip, width }: { tip: Tip | null; width: number }) {
  if (!tip) return null;
  const left = tip.x > width - 200 ? tip.x - 12 : tip.x + 12;
  const transform = tip.x > width - 200 ? "translateX(-100%)" : undefined;
  return (
    <div className="tooltip" style={{ left, top: tip.y + 8, transform }}>
      {tip.body}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}>
          <i style={{ background: it.color, opacity: it.dashed ? 0.6 : 1 }} className={it.dashed ? "dashed" : ""} />
          {it.label}
        </span>
      ))}
    </div>
  );
}
