import { useEffect, useState } from "react";

// Result files are produced by `python -m experiments.evaluate`; their shapes are documented
// there. The UI reads them loosely typed and only touches the fields it renders.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any;

export interface Load<T> {
  data: T | null;
  error: string | null;
}

const cache = new Map<string, Promise<Json>>();

export function fetchJson(path: string): Promise<Json> {
  let p = cache.get(path);
  if (!p) {
    p = fetch(path).then(async (r) => {
      if (!r.ok) {
        let detail = `${r.status} ${r.statusText}`;
        try {
          detail = (await r.json()).detail ?? detail;
        } catch {
          /* not JSON */
        }
        throw new Error(detail);
      }
      return r.json();
    });
    p.catch(() => cache.delete(path));
    cache.set(path, p);
  }
  return p;
}

export function useJson(path: string | null): Load<Json> {
  const [state, setState] = useState<Load<Json>>({ data: null, error: null });
  useEffect(() => {
    if (!path) return;
    let live = true;
    setState({ data: null, error: null });
    fetchJson(path).then(
      (data) => live && setState({ data, error: null }),
      (e: Error) => live && setState({ data: null, error: e.message }),
    );
    return () => {
      live = false;
    };
  }, [path]);
  return state;
}

// ---------------------------------------------------------------- formatting

export function fmt(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtSigned(x: number, digits = 0): string {
  return (x > 0 ? "+" : "") + fmt(x, digits);
}

export function pct(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined) return "–";
  return `${fmt(100 * x, digits)}%`;
}

export function fmtP(p: number): string {
  return p < 0.001 ? "<0.001" : p.toFixed(3);
}

export function digitsFor(unit: string): number {
  return unit === "fraction" ? 3 : unit === "trains" || unit === "count" ? 1 : 0;
}

export function clock(min: number): string {
  const h = Math.floor(min / 60);
  const m = Math.floor(min % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

// ---------------------------------------------------------------- shared palette

export const POLICY_COLORS: Record<string, string> = {
  iql: "var(--c-iql)",
  sarsa: "var(--c-sarsa)",
  fcfs: "var(--c-fcfs)",
  passenger_first: "var(--c-pf)",
  random: "var(--c-random)",
};

export const ACTION_NAMES = ["PROCEED", "LOOP", "HOLD"];
export const ACTION_COLORS = ["var(--c-proceed)", "var(--c-loop)", "var(--c-hold)"];
