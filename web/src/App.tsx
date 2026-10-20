import { useEffect, useState } from "react";
import { Behaviour } from "./pages/Behaviour";
import { Results } from "./pages/Results";
import { Safety } from "./pages/Safety";
import { Simulate } from "./pages/Simulate";
import { PageGate, ScenarioBar } from "./ScenarioBar";
import { scenarioQuery, ScenarioProvider, StatusProvider, usePage, useScenario } from "./scenario";

const TABS = [
  ["simulate", "Simulate"],
  ["safety", "Safety & Interlocking"],
  ["results", "Results"],
  ["behaviour", "Behaviour"],
] as const;
type Tab = (typeof TABS)[number][0];

function currentTab(): Tab {
  const raw = window.location.hash.replace("#/", "").replace("#", "").split("?")[0];
  if (raw === "replay" || raw === "overview") return "simulate";
  if (raw === "ablations" || raw === "learning" || raw === "sensitivity") return "results";
  return (TABS.find(([k]) => k === raw)?.[0] ?? "simulate") as Tab;
}

function Pages({ tab }: { tab: Tab }) {
  const ov = usePage("/api/page/overview");
  const { state } = useScenario();
  const agents = state.agents;
  const a = `agents=${agents}`;
  const ev = usePage(tab === "results" ? `/api/page/eval?${a}` : null);
  const beh = usePage(tab === "behaviour" ? `/api/page/behavior?${a}` : null);

  switch (tab) {
    case "simulate":
      return <PageGate pages={[ov]}>{() => <Simulate ov={ov.data} />}</PageGate>;
    case "safety":
      return <PageGate pages={[ov]}>{() => <Safety ov={ov.data} />}</PageGate>;
    case "results":
      return (
        <PageGate pages={[ev]}>
          {() => <Results ev={ev.data} />}
        </PageGate>
      );
    case "behaviour":
      return <PageGate pages={[beh]}>{() => <Behaviour beh={beh.data} />}</PageGate>;
  }
}

function Shell() {
  const [tab, setTab] = useState<Tab>(currentTab);
  const { state } = useScenario();
  const q = scenarioQuery(state, true);
  useEffect(() => {
    const on = () => setTab(currentTab());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="brand-mark" aria-hidden>
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M3 15h18" />
                <path d="M6 15c2 0 3-5 6-5s4 5 6 5" />
              </svg>
            </span>
            <div>
              <h1>Corridor Precedence Control</h1>
            </div>
          </div>
          <nav aria-label="Sections">
            {TABS.map(([k, label]) => (
              <a key={k} href={`#/${k}${q ? `?${q}` : ""}`} className={k === tab ? "on" : ""}>{label}</a>
            ))}
          </nav>
        </div>
      </header>
      <div className="app">
        <ScenarioBar />
        <main>
          <Pages tab={tab} />
        </main>
        <footer>Corridor Precedence Control · SIL-4 Interlocking Telemetry Active</footer>
      </div>
    </>
  );
}

export function App() {
  return (
    <ScenarioProvider>
      <StatusProvider>
        <Shell />
      </StatusProvider>
    </ScenarioProvider>
  );
}

export default App;
