import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import type { Json } from "./api";

// ---------------------------------------------------------------- scenario inputs
// Defaults and ranges mirror api/scenario.py (LIMITS). The default scenario is config.yaml.

export interface Field {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  percent?: boolean;
  help?: string;
}

export const TRAFFIC_DEFAULTS: Record<string, number> = {
  load: 1.35, passenger_share: 0.6, mix_vb: 0.2, mix_raj: 0.3, mix_exp: 0.5,
  late_prob: 0.3, late_mean_min: 12, jitter_min: 4, w_p: 10, w_f: 1,
};
export const DAY_DEFAULTS: Record<string, number> = { seed: 1, model_run: 0 };

export const GROUPS: { title: string; fields: Field[] }[] = [
  {
    title: "Traffic",
    fields: [
      { key: "load", label: "Traffic load", min: 0.6, max: 1.9, step: 0.05, percent: true,
        help: "Busiest-section utilisation the days are built to (±15%). Above 100% trains must overtake to run." },
      { key: "passenger_share", label: "Passenger share", min: 0.1, max: 0.9, step: 0.05, percent: true },
      { key: "mix_vb", label: "Vande Bharat weight", min: 0, max: 1, step: 0.05 },
      { key: "mix_raj", label: "Rajdhani weight", min: 0, max: 1, step: 0.05 },
      { key: "mix_exp", label: "Express weight", min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    title: "Disruption",
    fields: [
      { key: "late_prob", label: "Trains arriving late", min: 0, max: 1, step: 0.05, percent: true },
      { key: "late_mean_min", label: "Average lateness", min: 0, max: 60, step: 1, unit: "min" },
      { key: "jitter_min", label: "Departure jitter", min: 0, max: 20, step: 1, unit: "min",
        help: "Every train enters up to this many minutes after its slot." },
    ],
  },
  {
    title: "Scoring",
    fields: [
      { key: "w_p", label: "Passenger delay weight", min: 0, max: 50, step: 0.5,
        help: "Team reward = −(this × priority-weighted passenger delay + freight weight × freight delay). Agents trained on this scenario learn from it." },
      { key: "w_f", label: "Freight delay weight", min: 0, max: 10, step: 0.5 },
    ],
  },
];

export function showValue(f: Field, v: number): string {
  if (f.percent) return `${Math.round(v * 100)}%`;
  return `${+v.toFixed(2)}${f.unit ? ` ${f.unit}` : ""}`;
}

export type Agents = "default" | "trained";

export interface ScenarioState {
  traffic: Record<string, number>;
  day: Record<string, number>;
  agents: Agents;
}

function parseHash(): ScenarioState {
  const q = window.location.hash.split("?")[1] ?? "";
  const p = new URLSearchParams(q);
  const traffic = { ...TRAFFIC_DEFAULTS };
  const day = { ...DAY_DEFAULTS };
  p.forEach((v, k) => {
    if (v === "" || !Number.isFinite(+v)) return;
    if (k in traffic) traffic[k] = +v;
    if (k in day) day[k] = +v;
  });
  return { traffic, day, agents: p.get("agents") === "trained" ? "trained" : "default" };
}

/** Query string for the scenario (only non-default values, so the default scenario has a clean URL). */
export function scenarioQuery(s: ScenarioState, withDay = false): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(s.traffic)) if (v !== TRAFFIC_DEFAULTS[k]) p.set(k, String(v));
  if (withDay) for (const [k, v] of Object.entries(s.day)) if (v !== DAY_DEFAULTS[k]) p.set(k, String(v));
  if (s.agents === "trained") p.set("agents", "trained");
  return p.toString();
}

export function isDefaultTraffic(t: Record<string, number>): boolean {
  return Object.entries(TRAFFIC_DEFAULTS).every(([k, v]) => t[k] === v);
}

interface Ctx {
  state: ScenarioState;
  setTraffic: (t: Record<string, number>) => void;
  setDay: (d: Record<string, number>) => void;
  setAgents: (a: Agents) => void;
  /** Bumped whenever the server's state for this scenario may have changed (job finished). */
  version: number;
  bump: () => void;
}

const ScenarioContext = createContext<Ctx | null>(null);

export function ScenarioProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ScenarioState>(parseHash);
  const [version, setVersion] = useState(0);

  // keep the URL in sync so every tab and every shared link carries the scenario
  useEffect(() => {
    const rawTab = window.location.hash.replace("#/", "").split("?")[0];
    const tab = !rawTab || rawTab === "overview" ? "simulate" : rawTab;
    const q = scenarioQuery(state, true);
    const next = `#/${tab}${q ? `?${q}` : ""}`;
    if (window.location.hash !== next) window.history.replaceState(null, "", next);
  }, [state]);

  useEffect(() => {
    const on = () => {
      const s = parseHash();
      setState((cur) => (scenarioQuery(cur, true) === scenarioQuery(s, true) ? cur : s));
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const value: Ctx = {
    state,
    setTraffic: (traffic) => setState((s) => ({ ...s, traffic })),
    setDay: (day) => setState((s) => ({ ...s, day })),
    setAgents: (agents) => setState((s) => ({ ...s, agents })),
    version,
    bump: () => setVersion((v) => v + 1),
  };
  return <ScenarioContext.Provider value={value}>{children}</ScenarioContext.Provider>;
}

export function useScenario(): Ctx {
  const c = useContext(ScenarioContext);
  if (!c) throw new Error("useScenario outside ScenarioProvider");
  return c;
}

// ---------------------------------------------------------------- scenario-dependent data

export interface Page {
  data: Json | null;
  /** Background job computing this page (HTTP 202). */
  job: Json | null;
  queuePosition: number;
  /** The page needs agents trained on this scenario (response status "needs_training"). */
  needsTraining: boolean;
  error: string | null;
}

const done = new Map<string, Json>(); // finished pages are deterministic in their URL

/** Fetch an /api/page endpoint for the current scenario, polling while the server computes it. */
export function usePage(path: string | null): Page {
  const { state, version } = useScenario();
  const q = scenarioQuery({ ...state, agents: "default" }); // the page path carries the agents choice
  const url = path ? `${path}${path.includes("?") ? "&" : "?"}${q}` : null;
  const [page, setPage] = useState<Page>({ data: null, job: null, queuePosition: 0, needsTraining: false, error: null });
  const timer = useRef<number>(0);
  const shownUrl = useRef<string | null>(null);

  useEffect(() => {
    if (!url) return;
    let live = true;
    const cached = done.get(url);
    if (cached) {
      shownUrl.current = url;
      setPage({ data: cached, job: null, queuePosition: 0, needsTraining: false, error: null });
      return;
    }
    // a new page starts empty; re-checking the same page (after a job finished) keeps what is on screen
    if (shownUrl.current !== url) {
      shownUrl.current = url;
      setPage({ data: null, job: null, queuePosition: 0, needsTraining: false, error: null });
    }
    const poll = async () => {
      try {
        const r = await fetch(url);
        const body = await r.json().catch(() => ({}));
        if (!live) return;
        if (r.status === 200 && body.status === "needs_training") {
          setPage({ data: null, job: null, queuePosition: 0, needsTraining: true, error: null });
        } else if (r.status === 200) {
          done.set(url, body);
          setPage({ data: body, job: null, queuePosition: 0, needsTraining: false, error: null });
        } else if (r.status === 202) {
          setPage({ data: null, job: body.job, queuePosition: body.queue_position ?? 0, needsTraining: false, error: null });
          timer.current = window.setTimeout(poll, 1000);
        } else {
          setPage({ data: null, job: null, queuePosition: 0, needsTraining: false, error: body.detail ?? `${r.status} ${r.statusText}` });
        }
      } catch (e) {
        if (live) setPage({ data: null, job: null, queuePosition: 0, needsTraining: false, error: (e as Error).message });
      }
    };
    poll();
    return () => {
      live = false;
      clearTimeout(timer.current);
    };
  }, [url, version]); // version: re-check once a background job (e.g. training) has finished

  return page;
}

const StatusContext = createContext<{ status: Json | null; refresh: () => void }>({ status: null, refresh: () => undefined });

/** Polls the scenario status once for the whole app; read it with useStatus(). */
export function StatusProvider({ children }: { children: ReactNode }) {
  const value = useScenarioStatus();
  return <StatusContext.Provider value={value}>{children}</StatusContext.Provider>;
}

export function useStatus() {
  return useContext(StatusContext);
}

/** Server-side status of the current scenario (trained agents, running jobs), refreshed while jobs run. */
function useScenarioStatus(): { status: Json | null; refresh: () => void } {
  const { state, bump } = useScenario();
  const q = scenarioQuery({ ...state, agents: "default" });
  const [status, setStatus] = useState<Json | null>(null);
  const [tick, setTick] = useState(0);
  const refresh = useCallback(() => setTick((t) => t + 1), []);
  const wasRunning = useRef(false);

  useEffect(() => {
    let live = true;
    let t = 0;
    const load = async () => {
      try {
        const r = await fetch(`/api/scenario${q ? `?${q}` : ""}`);
        const s = await r.json();
        if (!live) return;
        setStatus(s);
        const running = (s.jobs ?? []).length > 0;
        if (wasRunning.current && !running) bump();
        wasRunning.current = running;
        if (running) t = window.setTimeout(load, 1000);
      } catch {
        /* keep last status */
      }
    };
    load();
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [q, tick]); // eslint-disable-line react-hooks/exhaustive-deps

  return { status, refresh };
}

export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined) return "";
  if (s < 60) return `${Math.max(1, Math.round(s))} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - 60 * m)} s`;
}
