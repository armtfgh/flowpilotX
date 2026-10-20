import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, AlertTriangle, Archive, ArrowRight, Atom, Beaker, Bot,
  Box, Check, CheckCircle2, ChevronDown, ChevronRight, CircleGauge,
  ClipboardCheck, Clock3, Database, Download, ExternalLink, FileJson,
  FileText, FlaskConical, Gauge, GitBranch, History, Image, Info,
  Layers3, Library, ListChecks, Menu, PackageCheck, Play, Plus,
  RefreshCw, Save, Send, Settings2, ShieldCheck, SlidersHorizontal,
  Sparkles, TestTube2, Thermometer, Trash2, Upload, User, Waypoints,
  X, Zap, ZoomIn, ZoomOut, Maximize2
} from "lucide-react";
import "./styles.css";
import { ProcessSummary, StageOverview, EngineeringHistory, Responses, CouncilTranscript, ScientificAssessment, displayUnits } from "./result_views";
import { InventoryWorkspace } from "./inventory_workspace";

declare const __FLOWPILOT_UI_BUILD__: string;
const WORKSPACE_STORAGE_KEY = "flowpilot_workspace_v1";

function restoredWorkspace(): Record<string, any> {
  try { return JSON.parse(sessionStorage.getItem(WORKSPACE_STORAGE_KEY) || "{}"); }
  catch { return {}; }
}

function setRunLocation(kind?: "job" | "run", id?: string) {
  const url = new URL(window.location.href);
  url.searchParams.delete("job"); url.searchParams.delete("run");
  if (kind && id) url.searchParams.set(kind, id);
  window.history.replaceState(null, "", url);
}

type Json = Record<string, any>;
type View = "design" | "inventory" | "runs" | "system";
type ResultTab = "overview" | "summary" | "topology" | "engineering" | "responses" | "recipe" | "inventory" | "council" | "cycles" | "json";

type Health = {
  status: string;
  version: string;
  corpus_records: number;
  inventory_profiles: number;
  models: Record<string, string>;
};

type Question = {
  question_id: string;
  section: string;
  question: string;
  expected_format: string;
  required: boolean;
  why_needed?: string;
  origin?: "core" | "conditional";
  decision_impact?: string;
  allow_unavailable?: boolean;
};

type Job = {
  job_id: string;
  archive_run_id?: string;
  status: string;
  progress: number;
  phase: string;
  messages: { time: string; level: string; message: string }[];
  result?: Json;
  error?: string | null;
  autosave_dir?: string | null;
  created_at?: string;
};

type InventoryProfile = {
  profile_id: string;
  name: string;
  laboratory?: string;
  version?: number;
  status?: string;
  lab_inventory: Json;
  operating_constraints?: Json;
  validation?: { valid: boolean; errors: string[]; warnings: string[]; unresolved_fields: string[] };
  [key: string]: any;
};

type ModelRoute = {
  route_id: string;
  label: string;
  provider: string;
  model: string;
  base_url?: string | null;
  local: boolean;
  available: boolean;
  availability: string;
  reason?: string;
};

type ModelCatalog = {
  models: ModelRoute[];
  defaults: { upstream: string; downstream: string };
};

const DEMO_PROTOCOL = `Fmoc-L-methionine (3.7 mmol, 1.0 equiv) and a flavin photocatalyst (10 mol%) were dissolved in acetonitrile (0.10 M). Air was supplied as the oxidant. The mixture was irradiated with a 420 nm LED at 21 °C for 5 min and afforded the product in 99% yield.`;

const DEMO_RESULT: Json = {
  confidence: "HIGH",
  recommended_disposition: "SCREEN",
  final_design: {
    schema_version: "flowpilot_final_design_v2.0",
    status: "executable",
    parameters: {
      concentration_M: 0.10, flow_rate_mL_min: 0.124,
      residence_time_min: 0.504796, residence_time_inlet_min: 0.504796,
      reactor_volume_mL: 1.0,
      tubing_ID_mm: 0.8, reactor_type: "PFA photocoil",
      material: "PFA", temperature_C: 21, BPR_bar: 3,
      wavelength_nm: 420, inventory_selection: "KHU photochemical system"
    },
    streams: [
      { stream_label: "A", phase: "liquid", flow_rate_mL_min: 0.124, contents: ["substrate", "photocatalyst in MeCN"] },
      { stream_label: "B", phase: "gas", gas_flow_sccm: 1.857, gas_flow_actual_mL_min: 0.505, contents: "air" }
    ],
    stages: [{ stage_number: 1, stage_name: "photochemical aerobic oxidation", reactor_volume_mL: 1.0, residence_time_inlet_min: 0.504796, residence_time_min: 0.504796 }],
    consistency: { passed: true, issues: [] }
  },
  process_topology: {
    unit_operations: [
      { op_id: "pump_a", label: "Liquid feed", op_type: "pump", inventory_item_id: "pump_a" },
      { op_id: "mfc_b", label: "Air MFC", op_type: "mfc", inventory_item_id: "mfc_b" },
      { op_id: "mix_1", label: "T-mixer", op_type: "mixer", inventory_item_id: "mix_1" },
      { op_id: "rx_1", label: "420 nm photocoil", op_type: "photoreactor", inventory_item_id: "rx_1" },
      { op_id: "bpr_1", label: "3 bar BPR", op_type: "bpr", inventory_item_id: "bpr_1" },
      { op_id: "sep_1", label: "G-L separator", op_type: "separator", inventory_item_id: "sep_1" },
      { op_id: "col_1", label: "Amber collector", op_type: "collector", inventory_item_id: "col_1" }
    ]
  },
  result_report: {
    status: "executable", issues: [], responses: [],
    stages: [{number: 1, name: "Photochemical aerobic oxidation", reactor: "1 mL PFA photocoil", volume_mL: 1,
      liquid_flow_mL_min: 0.124, gas_flow_stp_mL_min: 1.857, residence_time_min: 0.504796,
      residence_basis: "inlet/STP apparent", temperature_C: 21, pressure_bar: 3, pressure_basis: "gauge", wavelength_nm: 420, tubing_ID_mm: 0.8, material: "PFA"}],
    streams: [{label: "A", introduction_stage: 1, phase: "liquid", contents: ["substrate", "photocatalyst in MeCN"], flow_mL_min: 0.124, flow_basis: "liquid"},
      {label: "B", introduction_stage: 1, phase: "gas", contents: ["air"], flow_mL_min: 1.857, flow_basis: "inlet/STP"}]
  },
  chemistry_plan: {
    reaction_name: "Photocatalytic aerobic oxidation of Fmoc-L-methionine",
    mechanism_type: "flavin-mediated aerobic photooxidation",
    key_risks: ["oxygen transfer", "photon attenuation", "gas-liquid stability"],
    stream_logic: ["Substrate and photocatalyst share the liquid feed", "Air is metered independently before mixing"]
  },
  instrument_manifest: [
    { equipment_id: "pump_hplc_01", name: "HPLC liquid pump", role: "liquid feed" },
    { equipment_id: "mfc_air_01", name: "Air mass-flow controller", role: "gas feed" },
    { equipment_id: "photo_1ml_420", name: "1.0 mL PFA photocoil", role: "reaction" },
    { equipment_id: "bpr_3bar", name: "3 bar BPR", role: "pressure control" }
  ],
  council_rounds: 2,
  council_messages: [
    { agent: "Chemistry", content: "Transformation and oxygen demand are consistent with an aerobic photooxidation screen." },
    { agent: "Fluidics", content: "Demonstration data only: inlet/STP time closes against the 1.0 mL coil." },
    { agent: "Safety", content: "Use a vented separator and shield the illuminated pressurized coil." }
  ]
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string> || {}) };
  headers["X-FlowPilot-Client-Build"] = __FLOWPILOT_UI_BUILD__;
  if (!(init?.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch { /* retain status */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function formatValue(value: any, digits = 3): string {
  if (value === null || value === undefined || value === "") return "--";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  return String(value);
}

function shortTime(value?: string): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function operationId(op: Json, fallback = "unassigned"): string {
  return op.equipment_id || op.inventory_item_id || op.parameters?.inventory_item_id ||
    op.parameters?.inventory_equipment_id || op.operation_id || op.op_id || fallback;
}

function operationTitle(op: Json): string {
  const label = String(op.label || op.name || op.op_type || "operation");
  return label.split("|")[0].split("—")[0].trim() || label;
}

function streamContents(stream: Json): string {
  const value = stream.composition ?? stream.contents ?? stream.role;
  return Array.isArray(value) ? value.join("; ") : value || "--";
}

function Kpi({ label, value, note, icon, tone = "teal" }: { label: string; value: string; note: string; icon: React.ReactNode; tone?: string }) {
  return <div className={`kpi tone-${tone}`}>
    <div className="kpiHead"><span>{icon}{label}</span></div>
    <strong>{value}</strong><small>{note}</small>
  </div>;
}

function App() {
  const demo = new URLSearchParams(window.location.search).get("demo") === "1";
  const [saved] = useState<Json>(() => demo ? {} : restoredWorkspace());
  const [runtime, setRuntime] = useState<Json | null>(null);
  const [runtimeError, setRuntimeError] = useState("");
  const [storageError, setStorageError] = useState(false);
  const [restoring, setRestoring] = useState(!demo && /[?&](job|run)=/.test(window.location.search));
  const [view, setView] = useState<View>("design");
  const [mobileNav, setMobileNav] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [profiles, setProfiles] = useState<Json[]>([]);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog | null>(null);
  const [upstreamModelId, setUpstreamModelId] = useState<string>(saved.upstreamModelId || "");
  const [downstreamModelId, setDownstreamModelId] = useState<string>(saved.downstreamModelId || "");
  const [designPolicy, setDesignPolicy] = useState<string>(saved.designPolicy || "legacy");
  const [selectedProfileId, setSelectedProfileId] = useState<string>(saved.selectedProfile?.profile_id || "");
  const [selectedProfile, setSelectedProfile] = useState<InventoryProfile | null>(saved.selectedProfile || null);
  const [inventoryAlternatives, setInventoryAlternatives] = useState<Json[]>([]);
  const [protocol, setProtocol] = useState(demo ? DEMO_PROTOCOL : saved.protocol || "");
  const [intakePackage, setIntakePackage] = useState<Json | null>(demo ? {
    raw_protocol: DEMO_PROTOCOL, objective: "Produce one conservative inventory-constrained first flow screen.",
    ready_for_design: true, missing_question_ids: [], historical_data: null,
    hypotheses: ["oxygen transfer may control the practical rate"], operating_limits: { max_pressure_bar: 3 },
    extracted_batch_fields: { solvent: "acetonitrile", concentration_M: 0.1, reaction_time_min: 5, temperature_C: 21 }
  } : saved.intakePackage || null);
  const [questions, setQuestions] = useState<Question[]>(saved.questions || []);
  const [answers, setAnswers] = useState<Record<string, string>>(saved.answers || {});
  const [unavailable, setUnavailable] = useState<Record<string, boolean>>(saved.unavailable || {});
  const [useLlm, setUseLlm] = useState(saved.useLlm ?? true);
  const [job, setJob] = useState<Job | null>(demo ? { job_id: "demo", status: "completed", progress: 1, phase: "Design complete", messages: [], result: DEMO_RESULT } : null);
  const [result, setResult] = useState<Json | null>(demo ? DEMO_RESULT : null);
  const [resultTab, setResultTab] = useState<ResultTab>("overview");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [runs, setRuns] = useState<Json[]>([]);
  const [activeJobs, setActiveJobs] = useState<Job[]>([]);
  const deploymentBlocked = !demo && (!runtime || runtime.backend_stale || runtime.frontend_build_id !== __FLOWPILOT_UI_BUILD__);

  useEffect(() => {
    if (demo) return;
    let active = true;
    const check = async () => {
      try {
        const next = await api<Json>("/api/runtime", { cache: "no-store" });
        if (active) { setRuntime(next); setRuntimeError(""); }
      } catch { if (active) { setRuntime(null); setRuntimeError("The running server cannot verify this workspace version."); } }
    };
    check();
    const timer = window.setInterval(check, 30000);
    window.addEventListener("focus", check);
    return () => { active = false; clearInterval(timer); window.removeEventListener("focus", check); };
  }, []);
  useEffect(() => {
    if (demo || restoring) return;
    try {
      sessionStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify({
        protocol, intakePackage, questions, answers, unavailable, selectedProfile, upstreamModelId, downstreamModelId, useLlm, designPolicy
      }));
      setStorageError(false);
    } catch { setStorageError(true); }
  }, [protocol, intakePackage, questions, answers, unavailable, selectedProfile, upstreamModelId, downstreamModelId, useLlm, restoring, designPolicy]);

  const refreshSystem = async () => {
    try {
      const [h, p] = await Promise.all([
        api<Health>("/api/health"),
        api<{ profiles: Json[] }>("/api/inventory/profiles")
      ]);
      setHealth(h); setProfiles(p.profiles);
      const m = await api<ModelCatalog>("/api/models").catch(() => null);
      if (!m) {
        setModelCatalog(null);
        setError("The running backend is older than this GUI. Restart FlowPilot to enable model routing.");
        return;
      }
      setModelCatalog(m);
      setUpstreamModelId((current) => current || m.defaults.upstream);
      setDownstreamModelId((current) => current || m.defaults.downstream);
    } catch (e) { setError(String((e as Error).message || e)); }
  };

  useEffect(() => { if (!demo) refreshSystem(); }, []);
  useEffect(() => {
    if (!job || !job.job_id || job.status === "completed" || job.status === "failed" || job.job_id === "demo") return;
    const timer = window.setInterval(async () => {
      try {
        const next = await api<Job>(`/api/design/jobs/${job.job_id}`);
        await acceptJob(next);
      } catch (e) { setError(String((e as Error).message || e)); }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job?.job_id, job?.status]);

  const acceptIntake = (response: Json) => {
    setIntakePackage(response.package); setQuestions(response.pending_questions || []);
    setInventoryAlternatives(response.alternatives || []);
  };
  const reviewPackage = async (pkg: Json, profile: InventoryProfile | null, plan?: Json) => {
    const response = await api<Json>("/api/inventory/review", { method: "POST", body: JSON.stringify({
      intake_package: pkg, inventory_profile: profile, chemistry_plan: plan
    }) });
    acceptIntake(response);
  };
  const acceptJob = async (next: Job) => {
    setJob(next);
    if (next.status === "completed" && next.result) {
      setResult(next.result); setResultTab("overview");
      setDesignPolicy(next.result.pipeline_runtime?.design_policy || "legacy");
      const pkg = next.result.intake_package;
      if (pkg) {
        setIntakePackage(pkg); setProtocol(pkg.raw_protocol || "");
        setSelectedProfile(pkg.inventory_profile_snapshot || null);
        setSelectedProfileId(pkg.inventory_profile_snapshot?.profile_id || "");
        if (next.result.final_design?.status !== "executable") {
          await reviewPackage(pkg, pkg.inventory_profile_snapshot, next.result.chemistry_plan);
        }
      }
    }
    if (next.status === "failed") setError(next.error || "Design failed");
  };
  const applyProfile = async (profile: InventoryProfile | null) => {
    setBusy(true); setError("");
    try {
      if (intakePackage) {
        const pkg = profile ? intakePackage : { ...intakePackage,
          inventory_profile_snapshot: {}, inventory_constraints: null, operating_limits: null,
          answers: (intakePackage.answers || []).filter((a: Json) => a.source !== "inventory_profile") };
        await reviewPackage(pkg, profile, result?.chemistry_plan);
      }
      setSelectedProfile(profile); setSelectedProfileId(profile?.profile_id || "");
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const chooseProfile = async (id: string) => {
    setBusy(true); setError("");
    try { await applyProfile(id ? await api<InventoryProfile>(`/api/inventory/profiles/${id}`) : null); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const resolveInventory = async (confirmation: Json) => {
    setBusy(true); setError("");
    try {
      const response = await api<Json>("/api/inventory/resolve", { method: "POST", body: JSON.stringify({
        intake_package: intakePackage, inventory_profile: selectedProfile || intakePackage?.inventory_profile_snapshot,
        chemistry_plan: result?.chemistry_plan, ...confirmation
      }) });
      acceptIntake(response); setSelectedProfile(response.profile); setSelectedProfileId(response.profile.profile_id);
      setProfiles((await api<{ profiles: Json[] }>("/api/inventory/profiles")).profiles);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const analyze = async (newAnswers: Json[] = []) => {
    if (!protocol.trim()) return;
    setBusy(true); setError("");
    try {
      const response = await api<Json>("/api/intake/analyze", {
        method: "POST",
        body: JSON.stringify({
          raw_protocol: protocol,
          existing_package: intakePackage,
          answers: newAnswers,
          use_llm: newAnswers.length ? false : useLlm,
          inventory_profile: selectedProfile,
          upstream_model_id: upstreamModelId || undefined
        })
      });
      acceptIntake(response);
      setAnswers({}); setUnavailable({});
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(false); }
  };

  const submitAnswers = async () => {
    const payload = questions.filter((q) => unavailable[q.question_id] || answers[q.question_id]?.trim()).map((q) => ({
      question_id: q.question_id,
      status: unavailable[q.question_id] ? "unavailable" : "answered",
      answer: unavailable[q.question_id] ? null : answers[q.question_id],
      source: "webapp"
    }));
    await analyze(payload);
  };

  const runDesign = async () => {
    if (!intakePackage?.ready_for_design || deploymentBlocked || restoring) return;
    setBusy(true); setError("");
    try {
      const next = await api<Job>("/api/design/jobs", {
        method: "POST",
        body: JSON.stringify({
          batch_input: protocol,
          intake_package: intakePackage,
          inventory_profile: selectedProfile,
          upstream_model_id: upstreamModelId || undefined,
          downstream_model_id: downstreamModelId || undefined,
          runtime_options: {design_policy: designPolicy, candidate_budget: 12}
        })
      });
      setJob(next); setResult(null);
      setRunLocation("job", next.job_id);
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(false); }
  };

  const loadRuns = async () => {
    try {
      setRuns((await api<{ runs: Json[] }>("/api/runs")).runs);
      setActiveJobs((await api<{ jobs: Job[] }>("/api/design/jobs")).jobs.filter((item) => ["queued", "running", "unknown"].includes(item.status)));
    }
    catch (e) { setError(String((e as Error).message || e)); }
  };
  const openJob = async (jobId: string) => {
    setBusy(true); setError("");
    try {
      const next = await api<Job>(`/api/design/jobs/${encodeURIComponent(jobId)}`);
      setResult(null); await acceptJob(next);
      setView("design"); setRunLocation("job", jobId);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const openRun = async (runId: string) => {
    setBusy(true); setError("");
    try {
      const archived = await api<Json>(`/api/runs/${runId}`);
      setResult(archived);
      setDesignPolicy(archived.pipeline_runtime?.design_policy || "legacy");
      const archivedIntake = archived.intake_package || archived.intake_context || null;
      setIntakePackage(archivedIntake);
      if (archivedIntake?.raw_protocol) setProtocol(archivedIntake.raw_protocol);
      const snapshot = archivedIntake?.inventory_profile_snapshot || null;
      setSelectedProfile(snapshot); setSelectedProfileId(snapshot?.profile_id || "");
      if (archivedIntake && archived.final_design?.status !== "executable") {
        await reviewPackage(archivedIntake, snapshot, archived.chemistry_plan);
      }
      setJob({ job_id: `archive-${runId}`, archive_run_id: runId, status: "completed", progress: 1, phase: "Archived design", messages: [], result: archived });
      setResultTab("overview"); setView("design");
      setRunLocation("run", runId);
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(false); }
  };
  useEffect(() => {
    if (view !== "runs") return;
    loadRuns();
    const timer = window.setInterval(loadRuns, 10000);
    return () => window.clearInterval(timer);
  }, [view]);
  useEffect(() => {
    if (demo) return;
    const query = new URLSearchParams(window.location.search);
    const jobId = query.get("job"), runId = query.get("run");
    const restore = async () => {
      try {
        if (jobId) await acceptJob(await api<Job>(`/api/design/jobs/${encodeURIComponent(jobId)}`));
        else if (runId) await openRun(runId);
        else if (saved.intakePackage) await reviewPackage(saved.intakePackage, saved.selectedProfile);
      } catch (e) { setError((e as Error).message); }
      finally { setRestoring(false); }
    };
    restore();
  }, []);

  const nav = (next: View) => { setView(next); setMobileNav(false); };
  return <div className="appShell">
    <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
      <div className="brand"><div className="brandMark"><Waypoints size={19}/></div><div><h1>FlowPilot</h1><p>Batch-to-flow design</p></div></div>
      <nav className="nav">
        <button className={view === "design" ? "active" : ""} onClick={() => nav("design")}><FlaskConical size={17}/>Design studio<ChevronRight size={14}/></button>
        <button className={view === "inventory" ? "active" : ""} onClick={() => nav("inventory")}><Archive size={17}/>Inventory<ChevronRight size={14}/></button>
        <button className={view === "runs" ? "active" : ""} onClick={() => nav("runs")}><History size={17}/>Saved runs<ChevronRight size={14}/></button>
        <div className="navDivider"><span>System</span></div>
        <button className={view === "system" ? "active" : ""} onClick={() => nav("system")}><CircleGauge size={17}/>Status<ChevronRight size={14}/></button>
      </nav>
      <div className="sideStatus">
        <div><span className={`liveDot ${health || runtime || demo ? "ok" : ""}`}/><b>{health || runtime || demo ? "Core online" : "Connecting"}</b></div>
        <p>{result?.pipeline_runtime?.model_routing?.upstream_model ? `Upstream: ${result.pipeline_runtime.model_routing.upstream_model}` : "Backend connected"}</p>
        {result?.pipeline_runtime?.model_routing?.downstream_model && <p>Downstream: {result.pipeline_runtime.model_routing.downstream_model}</p>}
      </div>
      <div className="sideFooter"><ShieldCheck size={14}/>Deterministic gates authoritative</div>
    </aside>

    <main className="workspace">
      <header className="topbar">
        <button className="iconButton mobileMenu" onClick={() => setMobileNav(!mobileNav)} aria-label="Menu"><Menu size={19}/></button>
        <div><h1>{view === "design" ? "Design studio" : view === "inventory" ? "Laboratory inventory" : view === "runs" ? "Saved designs" : "System status"}</h1>
          <p>{view === "design" ? "Standardized intake, constrained design, and experimental refinement" : view === "inventory" ? "Convert laboratory documents into enforceable equipment profiles" : view === "runs" ? "Immutable design records and provenance" : "Models, corpus, and local services"}</p></div>
        <div className="topbarRight"><span className="version">v{health?.version || "2.0"}</span><button className="iconButton" onClick={refreshSystem} title="Refresh"><RefreshCw size={16}/></button></div>
      </header>

      {!demo && <div className="workspaceIdentity"><span>{window.location.host}</span><code>UI {__FLOWPILOT_UI_BUILD__}</code>
        <code>Server {runtime?.backend_build_id || "unverified"}</code><code>{job?.archive_run_id ? `Run ${job.archive_run_id}` : job ? `Job ${job.job_id}` : "No active job"}</code></div>}
      {deploymentBlocked && (runtime || runtimeError) && <div className="deploymentBanner" role="status"><AlertTriangle size={18}/><span>{runtimeError || (runtime?.backend_stale ? "Server update pending. Active jobs are not cancelled; new submissions are paused." : "Workspace update available. This open page is an older build.")}</span><button disabled={storageError} onClick={() => window.location.reload()}><RefreshCw size={15}/>Reload workspace</button></div>}
      {storageError && <div className="errorBanner">Browser draft storage is unavailable. Export your intake before reloading.</div>}

      {error && <div className="errorBanner"><AlertTriangle size={17}/><span>{error}</span><button onClick={() => setError("")}><X size={15}/></button></div>}
      {view === "design" && <DesignStudio
        protocol={protocol} setProtocol={setProtocol} intakePackage={intakePackage}
        questions={questions} answers={answers} setAnswers={setAnswers}
        unavailable={unavailable} setUnavailable={setUnavailable} useLlm={useLlm}
        setUseLlm={setUseLlm} profiles={profiles} selectedProfileId={selectedProfileId}
        setSelectedProfileId={chooseProfile} selectedProfile={selectedProfile}
        inventoryAlternatives={inventoryAlternatives} resolveInventory={resolveInventory}
        editIntake={() => { setResult(null); setJob(null); setResultTab("overview"); setRunLocation(); }}
        modelCatalog={modelCatalog} upstreamModelId={upstreamModelId}
        setUpstreamModelId={setUpstreamModelId} downstreamModelId={downstreamModelId}
        setDownstreamModelId={setDownstreamModelId}
        designPolicy={designPolicy} setDesignPolicy={setDesignPolicy}
        setIntakePackage={setIntakePackage}
        analyze={() => analyze()} submitAnswers={submitAnswers} runDesign={runDesign}
        busy={busy || restoring || deploymentBlocked} job={job} result={result} resultTab={resultTab} setResultTab={setResultTab}
      />}
      {view === "inventory" && <InventoryWorkspace api={api} profiles={profiles} refresh={refreshSystem} onUse={(profile) => { applyProfile(profile); setView("design"); }}/>}
      {view === "runs" && <div className="viewStack">
        {!!activeJobs.length && <section className="activeJobList"><h2>In-progress designs</h2>{activeJobs.map((item) => <div key={item.job_id}>
          <span><code>{item.job_id}</code><b>{item.phase}</b><small>{shortTime(item.created_at)}</small></span>
          <button onClick={() => openJob(item.job_id)}><Activity size={15}/>Open current job</button>
        </div>)}</section>}
        <RunsView runs={runs} onOpen={openRun}/>
      </div>}
      {view === "system" && <SystemView health={health}/>} 
    </main>
  </div>;
}

function DesignStudio(props: any) {
  const { protocol, setProtocol, intakePackage, setIntakePackage, questions, answers, setAnswers, unavailable, setUnavailable,
    useLlm, setUseLlm, profiles, selectedProfileId, setSelectedProfileId, selectedProfile,
    modelCatalog, upstreamModelId, setUpstreamModelId, downstreamModelId, setDownstreamModelId, designPolicy, setDesignPolicy,
    analyze, submitAnswers, runDesign, busy, job, result, resultTab, setResultTab,
    inventoryAlternatives, resolveInventory, editIntake } = props;
  const upstreamRoute = modelCatalog?.models.find((model: ModelRoute) => model.route_id === upstreamModelId);
  const downstreamRoute = modelCatalog?.models.find((model: ModelRoute) => model.route_id === downstreamModelId);
  const modelsReady = Boolean(upstreamRoute?.available && downstreamRoute?.available);
  const running = Boolean(job && ["queued", "running", "unknown"].includes(job.status));
  const intakeReady = Boolean(intakePackage?.ready_for_design && protocol === intakePackage.raw_protocol);
  const ready = intakeReady && modelsReady && !running;
  const review = intakePackage?.inventory_review;
  const withheld = result && result.final_design?.status !== "executable";
  const answered = intakePackage?.answers?.length || 0;
  const hasEvidence = Boolean(intakePackage?.historical_data && String(intakePackage.historical_data).trim());
  const inventoryBound = selectedProfile || intakePackage?.inventory_profile_snapshot || intakePackage?.inventory_constraints;
  const profileChoices = profiles.map((item: Json) => item.profile_id === selectedProfile?.profile_id ? {...item, ...selectedProfile} : item);
  if (selectedProfile && !profileChoices.some((item: Json) => item.profile_id === selectedProfile.profile_id)) profileChoices.push(selectedProfile);
  return <div className="viewStack">
    {!running && <section className="kpiGrid">
      <Kpi label="Intake" value={intakeReady ? "READY" : intakePackage ? "OPEN" : "NEW"} note={intakeReady ? "package frozen" : `${questions.length + (review?.questions?.length || 0)} questions pending`} icon={<ClipboardCheck size={14}/>} tone={intakeReady ? "green" : "amber"}/>
      <Kpi label="Evidence" value={hasEvidence ? "MEASURED" : "NONE"} note="highest authority" icon={<TestTube2 size={14}/>} tone={hasEvidence ? "blue" : "gray"}/>
      <Kpi label="Inventory" value={inventoryBound ? "BOUND" : "OPEN"} note={selectedProfile?.name || intakePackage?.inventory_profile_snapshot?.name || (inventoryBound ? "frozen intake constraints" : "select a laboratory")} icon={<PackageCheck size={14}/>} tone={inventoryBound ? "green" : "amber"}/>
      <Kpi label="Design" value={withheld ? "REVIEW" : job?.status?.toUpperCase() || "IDLE"} note={job?.phase || "no active run"} icon={<Activity size={14}/>} tone={withheld ? "amber" : job?.status === "completed" ? "green" : job?.status === "failed" ? "red" : "teal"}/>
    </section>}

    {!result && !running && <section className="designGrid">
      <div className="panel intakePanel">
        <div className="panelHead"><div><span className="eyebrow">01 / protocol</span><h2><Bot size={17}/>Standardized intake</h2></div><span className="tag">{answered} answers stored</span></div>
        <div className="messageList compact">
          <div className="message agent"><div className="messageMeta"><Bot size={13}/>FlowPilot intake</div><p>Provide the batch protocol. I will extract stated facts and apply the versioned core and deterministic conditional question bank.</p></div>
          {protocol && intakePackage && <div className="message user"><div className="messageMeta"><User size={13}/>Chemist</div><p>{protocol}</p></div>}
        </div>
        <label className="field"><span>Initial batch protocol</span><textarea className="protocolInput" value={protocol} onChange={(e) => setProtocol(e.target.value)} placeholder="Paste the complete batch protocol…"/></label>
        <div className="inlineControls"><label className="toggle"><input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)}/><i/><span>LLM-assisted extraction</span></label><button className="primary" disabled={busy || !protocol.trim()} onClick={analyze}><Sparkles size={15}/>{intakePackage ? "Re-analyze" : "Analyze intake"}</button></div>
      </div>

      <div className="panel questionsPanel">
        <div className="panelHead"><div><span className="eyebrow">02 / clarify</span><h2><ListChecks size={17}/>Fixed + conditional questions</h2></div><span className={`statusPill ${ready ? "success" : "warning"}`}>{ready ? <Check size={13}/> : <Clock3 size={13}/>} {ready ? "complete" : `${questions.length} pending`}</span></div>
        {intakePackage?.question_set_hash && <div className="questionSetMeta"><span>{intakePackage.question_bank_version}</span><code>{intakePackage.question_set_hash.slice(0, 12)}</code></div>}
        {!intakePackage && <Empty icon={<Send size={22}/>} title="Analyze the protocol first" text="The reproducible question bank will appear here."/>}
        {intakePackage && !questions.length && <div className="successState"><CheckCircle2 size={30}/><h3>{intakePackage.ready_for_design ? "Design input is frozen" : "Protocol answers saved"}</h3><p>{review?.ready === false ? "Equipment requirements remain unresolved." : "All mandatory questions are answered or explicitly unavailable."}</p></div>}
        {!!questions.length && <div className="questionScroll">{questions.map((q: Question) => <div className="question" key={q.question_id}>
          <div className="questionHead"><code>{q.question_id}</code><span>{q.section.replaceAll("_", " ")} · {q.origin || "core"}</span></div><p>{q.question}</p>
          {q.decision_impact && <small className="questionImpact">Design impact: {q.decision_impact}</small>}
          <textarea disabled={busy || unavailable[q.question_id]} value={answers[q.question_id] || ""} onChange={(e) => setAnswers((current: Json) => ({...current, [q.question_id]: e.target.value}))} placeholder={q.expected_format}/>
          {q.allow_unavailable !== false && <label className="check"><input type="checkbox" disabled={busy} checked={Boolean(unavailable[q.question_id])} onChange={(e) => setUnavailable((current: Json) => ({...current, [q.question_id]: e.target.checked}))}/><span>Explicitly unavailable</span></label>}
        </div>)}</div>}
        {!!questions.length && <button className="primary full" disabled={busy || !questions.some((q: Question) => unavailable[q.question_id] || answers[q.question_id]?.trim())} onClick={submitAnswers}><CheckCircle2 size={15}/>Save answers</button>}
      </div>

      <div className="panel launchPanel">
        <div className="panelHead"><div><span className="eyebrow">03 / constrain</span><h2><Archive size={17}/>Laboratory context</h2></div></div>
        <div className="modelSelectors">
          <div className="modelSelectorsHead"><Bot size={14}/><span>Model routing</span></div>
          <label className="field"><span>Upstream chemistry</span><select aria-label="Upstream chemistry model" value={upstreamModelId} onChange={(e) => setUpstreamModelId(e.target.value)} disabled={!modelCatalog?.models.length}>{modelCatalog?.models.map((model: ModelRoute) => <option disabled={!model.available} value={model.route_id} key={`up-${model.route_id}`}>{model.label}{model.available ? "" : " · unavailable"}</option>)}</select><small>{upstreamRoute?.available ? "Protocol interpretation and chemistry plan" : upstreamRoute?.reason || "Select an available model"}</small></label>
          <label className="field"><span>Design policy</span><select aria-label="Design policy" value={designPolicy} onChange={(e) => setDesignPolicy(e.target.value)} disabled={running}><option value="legacy">Legacy</option><option value="scientific_v2">FlowPilot 2.0 · private scientific preview</option></select></label>
          {designPolicy === "scientific_v2" && intakePackage && <label className="field"><span>Screening priority</span><select aria-label="Screening priority" value={intakePackage.screening_priority || "auto"} disabled={running || busy} onChange={(e) => setIntakePackage((p: Json) => ({...p, screening_priority: e.target.value}))}><option value="auto">Interpret stated objective</option><option value="balanced">Balanced baseline screen</option><option value="yield_priority">Yield-focused screen</option><option value="throughput_priority">Throughput-focused screen</option></select></label>}
          <label className="field"><span>Downstream and council</span><select aria-label="Downstream and council model" value={downstreamModelId} onChange={(e) => setDownstreamModelId(e.target.value)} disabled={!modelCatalog?.models.length}>{modelCatalog?.models.map((model: ModelRoute) => <option disabled={!model.available} value={model.route_id} key={`down-${model.route_id}`}>{model.label}{model.available ? "" : " · unavailable"}</option>)}</select><small>{downstreamRoute?.available ? "Flow proposal, council agents, revision, and selection" : downstreamRoute?.reason || "Select an available model"}</small></label>
        </div>
        <label className="field"><span>Inventory profile</span><select aria-label="Inventory profile" disabled={busy || running} value={selectedProfileId} onChange={(e) => setSelectedProfileId(e.target.value)}><option value="">No saved profile selected</option>{profileChoices.map((p: Json) => <option value={p.profile_id} key={p.profile_id}>{p.name} · v{p.version}</option>)}</select></label>
        {selectedProfile ? <InventorySnapshot profile={selectedProfile}/> : <Empty icon={<Archive size={22}/>} title="Inventory not bound" text="Select a validated profile or mark inventory unavailable during intake."/>}
        <div className="authorityOrder"><span>Design authority</span><ol><li>Measured evidence</li><li>Hard constraints</li><li>Protocol facts</li><li>Chemist hypotheses</li><li>Model inference</li></ol></div>
        <button className="runButton" disabled={busy || !ready} onClick={runDesign}><Play size={17}/><span><b>Run FlowPilot design</b><small>{!intakePackage?.ready_for_design ? "Complete intake to unlock" : !modelsReady ? "Select available upstream and downstream models" : "Start constrained pipeline"}</small></span><ArrowRight size={16}/></button>
      </div>
    </section>}

    {withheld && <section className="inventoryRecovery">
      <div className="panelHead"><h2><PackageCheck size={17}/>Resolve laboratory requirements</h2><button disabled={busy} onClick={editIntake}><RefreshCw size={15}/>Return to intake</button></div>
      <label className="field"><span>Inventory profile</span><select aria-label="Inventory profile" disabled={busy} value={selectedProfileId} onChange={(e) => setSelectedProfileId(e.target.value)}><option value="">No saved profile selected</option>{profileChoices.map((p: Json) => <option key={p.profile_id} value={p.profile_id}>{p.name} · v{p.version}</option>)}</select></label>
      <p>Original run preserved. Current profile: {selectedProfile?.name || "intake inventory"}, version {selectedProfile?.version || "unsaved"}.</p>
      <button className="primary" disabled={busy || !ready} onClick={runDesign}><Play size={15}/>Run with resolved inventory</button>
    </section>}
    {intakePackage && !running && (!result || withheld) && <InventoryResolutionPanel review={review} alternatives={inventoryAlternatives}
      busy={busy || running} onChoose={setSelectedProfileId} onConfirm={resolveInventory}/>}

    {job && job.status !== "completed" && !result && <JobProgress job={job}/>} 
    {withheld && review?.ready && <h2 className="previousRunHeading">Previous run: original inventory</h2>}
    {result && <ResultWorkspace result={result} job={job} tab={resultTab} setTab={setResultTab}/>} 
  </div>;
}

function InventoryResolutionPanel({ review, alternatives, busy, onChoose, onConfirm }: {
  review?: Json; alternatives: Json[]; busy: boolean;
  onChoose: (id: string) => void; onConfirm: (data: Json) => void;
}) {
  if (!review || !review.requirements?.length) return null;
  return <section className="inventoryRecovery" aria-label="Equipment requirements">
    <div className="panelHead"><h2><PackageCheck size={17}/>Equipment requirements</h2>
      <span className={`statusPill ${review.ready ? "success" : "warning"}`}>{review.ready ? "Precheck passed" : review.status === "conceptual_only" ? "Required equipment unavailable" : "Confirmation needed"}</span></div>
    <p>{review.profile_name || "Intake inventory"}{review.profile_version ? ` · v${review.profile_version}` : ""}</p>
    <div className="requirementRows">{review.requirements.map((item: Json) => <div key={item.requirement_id}>
      <b>{item.category.replaceAll("_", " ")}</b><span>{item.available_count} available / {item.required_count} needed</span>
      <span className={item.status === "available" ? "requirementAvailable" : "requirementMissing"}>{item.status.replaceAll("_", " ")}</span>
    </div>)}</div>
    {!!review.assumed_standard_accessories?.length && <p className="questionImpact">Standard-accessory assumptions: {review.assumed_standard_accessories.map((item: Json) => item.category).join(", ")}. These are not confirmed equipment assignments.</p>}
    {(alternatives || []).map((profile: Json) => <div className="profileAlternative" key={profile.profile_id}>
      <span><b>{profile.name} · v{profile.version}</b><small>{profile.basis}</small></span>
      <button disabled={busy} onClick={() => onChoose(profile.profile_id)}><Archive size={15}/>Select profile</button>
    </div>)}
    {(review.questions || []).map((question: Json) => <EquipmentConfirmation key={`${review.input_sha256}-${question.question_id}`} question={question} busy={busy} onConfirm={onConfirm}/>)}
    {(review.confirmations || []).map((question: Json) => <details key={question.question_id}><summary>{question.title}: marked unavailable. Revise confirmation</summary>
      <EquipmentConfirmation question={question} busy={busy} onConfirm={onConfirm}/></details>)}
  </section>;
}

function EquipmentConfirmation({ question, busy, onConfirm }: { question: Json; busy: boolean; onConfirm: (data: Json) => void }) {
  const [status, setStatus] = useState("available");
  const [values, setValues] = useState<Json>({ quantity: "1" });
  const [note, setNote] = useState("");
  const fields = [{key: "equipment_id", label: "Equipment ID", type: "text"}, {key: "name", label: "Equipment name", type: "text"},
    {key: "quantity", label: "Quantity available", type: "integer"}, ...question.fields];
  return <form className="equipmentConfirmation" onSubmit={(event) => { event.preventDefault(); onConfirm({ category: question.category, status, equipment: values, note }); }}>
    <div className="questionHead"><code>{question.question_id}</code><b>{question.title}</b></div>
    <p>{question.question}</p><p>{question.reason}</p>
    <label className="field"><span>Equipment availability</span><select aria-label="Equipment availability" disabled={busy} value={status} onChange={(e) => setStatus(e.target.value)}>
      <option value="available">Available: confirm specifications</option><option value="unavailable">Unavailable in our laboratory</option></select></label>
    {status === "available" && <div className="confirmationFields">{fields.map((field: Json) => <label className="field" key={field.key}><span>{field.label}</span>
      <input disabled={busy} required type={["number", "integer"].includes(field.type) ? "number" : "text"}
        step={field.type === "integer" ? "1" : "any"} value={values[field.key] || ""}
        onChange={(e) => setValues({...values, [field.key]: e.target.value})}/></label>)}</div>}
    <label className="field"><span>Confirmation note or specification source</span><input disabled={busy} value={note} onChange={(e) => setNote(e.target.value)}/></label>
    <button className="primary" disabled={busy} type="submit"><Save size={15}/>Save inventory confirmation</button>
  </form>;
}

function JobProgress({ job }: { job: Job }) {
  return <section className="panel jobPanel"><div className="jobTitle"><div className="spinner"/><div><h2>{job.phase}</h2><p>Job {job.job_id} · FlowPilot is preserving every stage transition.</p></div><strong>{Math.round((job.progress || 0) * 100)}%</strong></div>
    <div className="progress"><i style={{width: `${Math.max(2, job.progress * 100)}%`}}/></div>
    <div className="logConsole">{job.messages?.slice(-8).map((m, i) => <div key={i}><span>{shortTime(m.time)}</span>{m.message}</div>)}</div></section>;
}

function ResultWorkspace({ result, job, tab, setTab }: { result: Json; job: Job | null; tab: ResultTab; setTab: (tab: ResultTab) => void }) {
  const final = result.final_design || {};
  const params = final.parameters || {};
  const executable = final.status === "executable";
  const flowReviewRequired = Boolean(final.flow_operability?.applicable);
  const assumedEquipment = (result.instrument_manifest || []).filter((item: Json) => item.requires_pre_run_verification);
  const pendingAssumptions = (result.design_realization?.decisions || []).filter((item: Json) => item.confirmation_required);
  const gasDelivery = result.scientific_assessment?.source_context?.gas_delivery || result.proposal?.scientific_design?.gas_delivery_policy;
  const tabs: [ResultTab, string, React.ReactNode][] = [
    ["overview", "Overview", <Gauge size={14}/>], ["topology", "Process", <GitBranch size={14}/>],
    ["summary", "Process summary", <ListChecks size={14}/>],
    ["engineering", "Engineering", <SlidersHorizontal size={14}/>], ["recipe", "Chemistry", <Beaker size={14}/>],
    ["inventory", "Equipment", <Archive size={14}/>], ["council", "Council", <Layers3 size={14}/>],
    ["responses", "Responses", <ClipboardCheck size={14}/>],
    ["cycles", "Experiment loop", <RefreshCw size={14}/>], ["json", "JSON", <FileJson size={14}/>]
  ];
  const download = () => {
    if (job?.archive_run_id) window.open(`/api/runs/${job.archive_run_id}/download`, "_blank");
    else if (job?.job_id && job.job_id !== "demo") window.open(`/api/design/jobs/${job.job_id}/download`, "_blank");
  };
  return <section className="resultWorkspace">
    <div className={`resultBanner ${!executable ? "blocked" : flowReviewRequired ? "review" : "executable"}`}><div>{executable && !flowReviewRequired ? <CheckCircle2 size={22}/> : <AlertTriangle size={22}/>}<span><b>{!executable ? "Design withheld" : flowReviewRequired ? "Screening proposal: laboratory review required" : "Executable screening design"}</b><small>{executable && flowReviewRequired ? "Parameters are available for review, not approved for execution. Resolve branch protection and operating-pressure checks first." : result.disposition_rationale || (executable ? "Canonical contract closed against inventory and engineering gates." : "Inspect blocking reasons before execution.")}</small></span></div><div className="bannerActions"><span className="confidence">{result.confidence || "--"} confidence</span><button onClick={download} disabled={job?.job_id === "demo"}><Download size={15}/>JSON</button></div></div>
    {result.final_design?.flow_operability?.applicable && <div className="inventoryRecovery" role="alert"><b>Flow operability requires laboratory review</b><p>Backflow paths and protective equipment remain to be verified. See the Council tab before using these proposed conditions.</p></div>}
    {!!assumedEquipment.length && <div className="warningRow"><AlertTriangle size={17}/><span><b>Equipment verification required before laboratory use</b>{assumedEquipment.map((item: Json) => item.name || item.equipment_id).join("; ")}</span></div>}
    {gasDelivery?.requires_chemist_confirmation && <div className="warningRow gasChangeWarning"><AlertTriangle size={18}/><span><b>Unapproved gas-feed change: {gasDelivery.batch_species || "batch gas"} to {gasDelivery.species}</b>Laboratory confirmation and safety review are required before use. {gasDelivery.rationale}</span></div>}
    {!!pendingAssumptions.length && <div className="warningRow"><AlertTriangle size={17}/><div><b>Chemist review required before laboratory use</b><ul>{pendingAssumptions.map((item: Json, i: number) => <li key={i}>{item.decision === "oxidant_gas_inventory_substitution"
      ? `Gas substitution: ${item.from_gas} to ${item.to_gas}. ${item.basis}`
      : item.decision === "component_quantity_screening_assumption"
        ? `${item.component}: ${item.loading_mol_pct != null ? `${item.loading_mol_pct} mol%` : `${item.selected_concentration_M} M`} is a screening assumption, not a measured or confirmed recipe.`
        : item.basis || item.decision.replaceAll("_", " ")}</li>)}</ul></div></div>}
    <div className="resultTabs">{tabs.map(([id, label, icon]) => <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>{icon}{label}</button>)}</div>
    <div className="resultBody">
      {tab === "overview" && <Overview result={result} params={params} executable={executable}/>} 
      {tab === "topology" && <Topology result={result} job={job}/>} 
      {tab === "summary" && <ProcessSummary result={result}/>}
      {tab === "responses" && <Responses result={result}/>}
      {tab === "engineering" && (executable ? <EngineeringHistory result={result}/> : <Overview result={result} params={{}} executable={false}/>)}
      {tab === "recipe" && <Chemistry result={result} final={final}/>} 
      {tab === "inventory" && <Equipment result={result}/>} 
      {tab === "council" && <CouncilTranscript result={result}/>}
      {tab === "cycles" && (result.pipeline_runtime?.design_policy === "scientific_v2" ? <Empty icon={<TestTube2 size={22}/>} title="New evidence requires council review" text="Attach measured results to a new standardized intake. The private preview does not apply legacy scalar refinement to a two-stage design."/> : executable ? <ExperimentLoop job={job} result={result}/> : <Empty icon={<TestTube2 size={22}/>} title="No executable design to refine" text="Resolve the design requirements before recording an experiment against this proposal."/>)}
      {tab === "json" && <JsonViewer value={result}/>} 
    </div>
  </section>;
}

function Overview({ result, params, executable }: { result: Json; params: Json; executable: boolean }) {
  if (!executable) return <div className="withheldSummary"><h2>{result.inventory_preflight ? "Equipment review required" : "Design validation requires review"}</h2>
    <p>{result.explanation || result.disposition_rationale}</p>
    {(result.inventory_preflight?.unresolved_requirements || result.inventory_allocation?.unresolved_requirements || []).map((item: Json, i: number) =>
      <div className="warningRow" key={i}><AlertTriangle size={17}/><span><b>{item.category?.replaceAll("_", " ") || item.operation_id}</b>{item.reason}</span></div>)}
    {(result.final_design?.consistency?.issues || []).map((issue: any, i: number) => <p key={i}>{typeof issue === "string" ? issue : issue.message || JSON.stringify(issue)}</p>)}
  </div>;
  const report = result.result_report || {};
  const gases = (report.streams || []).filter((s: Json) => s.phase === "gas");
  const hasGas = gases.length > 0;
  const stages = report.stages || [];
  if (stages.length > 1) return <><ScientificAssessment result={result}/><StageOverview result={result} closure={<Closure result={result}/>}/></>;
  const stage = stages[0] || {};
  const exact = (v: any) => v == null ? "--" : String(Number(Number(v).toPrecision(6)));
  const conditions = {...params, reactor_volume_mL: stage.volume_mL, flow_rate_mL_min: stage.liquid_flow_mL_min,
    temperature_C: stage.temperature_C, BPR_bar: stage.pressure_bar, wavelength_nm: stage.wavelength_nm,
    tubing_ID_mm: stage.tubing_ID_mm, material: stage.material, tubing_material: stage.material};
  return <div className="overviewGrid">
    <div className="metricBand">
      <Kpi label={hasGas ? "Residence time · inlet/STP" : "Residence time"} value={`${exact(stage.residence_time_min)} min`} note={stage.residence_basis || "not recorded"} icon={<Clock3 size={14}/>} tone="teal"/>
      <Kpi label="Liquid flow" value={`${exact(stage.liquid_flow_mL_min)} mL/min`} note={`${formatValue(params.concentration_M)} M feed`} icon={<Activity size={14}/>} tone="blue"/>
      <Kpi label="Reactor" value={`${exact(stage.volume_mL)} mL`} note={`${stage.reactor || stage.material || "--"} · ${exact(stage.tubing_ID_mm)} mm ID`} icon={<Box size={14}/>} tone="amber"/>
      <Kpi label="Gas at inlet/STP" value={hasGas ? `${exact(stage.gas_flow_stp_mL_min)} mL/min` : "No gas feed"} note={hasGas ? `${gases.map((s: Json) => `${s.label}: ${exact(s.equiv)} equiv`).join("; ")} · 273.15 K, 1.01325 bar` : "liquid-only process"} icon={<Zap size={14}/>} tone="green"/>
    </div>
    {(report.issues || []).map((issue: string) => <div className="warningRow" key={issue}>{issue}</div>)}
    <div className="panel unframed"><div className="panelHead"><h2><ClipboardCheck size={17}/>Run conditions</h2><span className={`statusPill ${executable ? "success" : "danger"}`}>{executable ? "authoritative" : "diagnostic"}</span></div><ConditionTable params={conditions}/></div>
    <div className="panel unframed"><div className="panelHead"><h2><ShieldCheck size={17}/>Closure</h2></div><Closure result={result}/></div>
  </div>;
}

function ConditionTable({ params, stages = [] }: { params: Json; stages?: Json[] }) {
  const condition = (key: string, value: any) => value !== null && value !== undefined ? formatValue(value)
    : stages.some((stage) => stage[key] !== null && stage[key] !== undefined)
      ? stages.map((stage, i) => `Stage ${stage.stage_number || i + 1}: ${formatValue(stage[key])}`).join("; ") : "--";
  const rows = [
    ["Concentration", formatValue(params.concentration_M), "M"], ["Temperature", condition("temperature_C", params.temperature_C), "°C"],
    ["Pressure", formatValue(params.BPR_bar), "bar"], ["Wavelength", condition("wavelength_nm", params.wavelength_nm), "nm"],
    ["Liquid flow", formatValue(params.flow_rate_mL_min), "mL/min"], ["Reactor volume", formatValue(params.reactor_volume_mL), "mL"],
    ["Tubing ID", condition("d_mm", params.tubing_ID_mm), "mm"], ["Material", condition("material", params.material || params.tubing_material), ""]
  ];
  return <div className="conditionTable">{rows.map(([label, value, unit]) => <div key={label}><span>{label}</span><b>{value} <small>{unit}</small></b></div>)}</div>;
}

function Closure({ result }: { result: Json }) {
  const issues = result.final_design?.consistency?.issues || [];
  const checks = [
    ["Final schema", Boolean(result.final_design?.schema_version)],
    ["Numerical closure", Boolean(result.final_design?.consistency?.passed)],
    ["Inventory assignment", Boolean(result.inventory_allocation?.complete ?? result.instrument_manifest?.length)],
    ["Process topology", Boolean(result.process_topology?.unit_operations?.length)],
    ["Autosave", Boolean(result.autosave_dir)]
  ];
  if (result.final_design?.flow_operability?.laboratory_execution_status === "review_required") {
    checks.push(["Flow operability", false]);
  }
  return <div className="checkList">{checks.map(([label, pass]) => <div key={String(label)} className={pass ? "pass" : "fail"}>{pass ? <CheckCircle2 size={16}/> : <AlertTriangle size={16}/>}<span>{label}</span><b>{pass ? "closed" : "review"}</b></div>)}{issues.slice(0, 4).map((issue: Json, i: number) => <div className="issue" key={i}><Info size={16}/><span>{issue.message || String(issue)}</span></div>)}</div>;
}

function Topology({ result, job }: { result: Json; job: Job | null }) {
  const executable = result.final_design?.status === "executable";
  const topology = executable ? result.process_topology : result.diagnostic_topology || result.process_requirements_topology;
  const ops = topology?.unit_operations || [];
  const artifact = job?.archive_run_id
    ? `/api/runs/${job.archive_run_id}/artifacts/${executable ? "process-svg" : "diagnostic-svg"}`
    : job?.job_id && job.job_id !== "demo" ? `/api/design/jobs/${job.job_id}/artifacts/${executable ? "process-svg" : "diagnostic-svg"}` : "";
  const [artifactFailed, setArtifactFailed] = useState(false);
  const [zoom, setZoom] = useState(1);
  useEffect(() => setArtifactFailed(false), [artifact]);
  useEffect(() => setZoom(1), [artifact]);
  return <div className="topologyLayout"><div className="topologyCanvas">
    <div className="topologyHead"><div><h2>{executable ? result.final_design?.flow_operability?.applicable ? "Proposed process topology" : "Executable process topology" : "Requirements topology"}</h2><p>{ops.length} declared unit operations · inlet/STP gas basis</p></div><div className="diagramTools">
      <button className="iconButton" title="Zoom out" aria-label="Zoom out" disabled={zoom <= 1} onClick={() => setZoom(z => Math.max(1, z - 0.5))}><ZoomOut size={16}/></button>
      <button className="iconButton" title="Zoom in" aria-label="Zoom in" disabled={zoom >= 5} onClick={() => setZoom(z => Math.min(5, z + 0.5))}><ZoomIn size={16}/></button>
      <button className="iconButton" title="Fit diagram" aria-label="Fit diagram" onClick={() => setZoom(1)}><Maximize2 size={16}/></button>
      {artifact && <a className="iconButton" title="Open full image" aria-label="Open full image" href={artifact} target="_blank"><ExternalLink size={16}/></a>}</div></div>
    {artifact && !artifactFailed ? <div className="diagramViewport"><div style={{width: `${zoom * 100}%`}}><img className="processImage" src={artifact} alt="FlowPilot process topology" onError={() => setArtifactFailed(true)}/></div></div>: null}
    {(!artifact || artifactFailed) && <ProcessChain operations={ops}/>} 
  </div><div className="operationList"><h3>Unit operations</h3>{ops.map((op: Json, i: number) => <div key={op.operation_id || op.op_id || i}><span>{String(i + 1).padStart(2, "0")}</span><div><b>{op.instrument_name || op.parameters?.instrument_name || operationTitle(op)}</b><small>{operationId(op)}</small></div></div>)}</div></div>;
}

function ProcessChain({ operations }: { operations: Json[] }) {
  if (!operations?.length) return <Empty icon={<GitBranch size={22}/>} title="No topology published" text="The deterministic process graph is unavailable for this result."/>;
  return <div className="processChain">{operations.map((op, i) => <React.Fragment key={op.operation_id || op.op_id || i}><div className={`processNode type-${op.op_type}`}><OperationIcon type={op.op_type}/><span>{operationTitle(op)}</span><small>{op.parameters?.volume_mL ? `${op.parameters.volume_mL} mL · ${operationId(op)}` : operationId(op)}</small></div>{i < operations.length - 1 && <ArrowRight size={17} className="chainArrow"/>}</React.Fragment>)}</div>;
}

function OperationIcon({ type }: { type: string }) {
  if (type?.includes("reactor") || type === "photoreactor") return <Atom size={19}/>;
  if (type === "pump" || type === "mfc") return <Activity size={19}/>;
  if (type === "separator") return <GitBranch size={19}/>;
  if (type === "collector") return <TestTube2 size={19}/>;
  if (type === "bpr") return <Gauge size={19}/>;
  return <Waypoints size={19}/>;
}


function Chemistry({ result, final }: { result: Json; final: Json }) {
  const plan = result.chemistry_plan || {};
  if (final.status !== "executable") return <div className="withheldSummary"><h2>Chemistry analysis</h2>
    <p>{plan.reaction_name || result.batch_record?.reaction_description || "No chemistry analysis stored."}</p>
    <p>{plan.mechanism_type}</p><TagList values={plan.key_risks || plan.safety_flags || []}/>
    <p>No accepted operating procedure is available for this withheld design.</p></div>;
  return <div className="splitView"><section><div className="sectionTitle"><h2>Chemistry plan</h2><span>upstream analysis</span></div><dl className="detailList"><dt>Reaction</dt><dd>{plan.reaction_name || result.batch_record?.reaction_description || "--"}</dd><dt>Mechanism</dt><dd>{plan.mechanism_type || "--"}</dd><dt>Solvent</dt><dd>{plan.solvent_rationale || result.batch_record?.solvent || "--"}</dd></dl><TagList values={plan.key_risks || plan.safety_flags || []}/>
    {!!final.stream_components?.length && <><div className="sectionTitle"><h2>Feed components</h2><span>final quantities</span></div><table className="dataTable"><thead><tr><th>Stream</th><th>Component</th><th>Quantity</th></tr></thead><tbody>{final.stream_components.map((item: Json, i: number) => <tr key={i}><td>{item.stream_label}</td><td>{item.name}</td><td>{item.quantification_required === false ? "Solvent" : <>{item.concentration_M != null ? `${item.concentration_M} M` : "Not specified"}{item.loading_mol_pct != null ? `; ${item.loading_mol_pct} mol%` : item.molar_equiv != null ? `; ${item.molar_equiv} equiv` : ""}</>}{item.provenance?.includes("screening_assumption_requires_confirmation") && <small>Unconfirmed screening assumption</small>}</td></tr>)}</tbody></table></>}
    </section><section><div className="sectionTitle"><h2>Operating procedure</h2><span>compiled from final design</span></div>{final.operating_procedure?.length ? <ol className="procedure">{final.operating_procedure.map((step: Json, i: number) => <li key={step.step_id || i}>{displayUnits(step.instruction)}</li>)}</ol> : <p>No compiled procedure is stored in this result.</p>}</section></div>;
}

function Equipment({ result }: { result: Json }) {
  const items = result.instrument_manifest || [];
  const unresolved = result.inventory_allocation?.unresolved_requirements || [];
  return <div><div className="sectionTitle"><h2>Assigned equipment</h2><span>{items.length} instruments</span></div>{items.length ? <div className="equipmentGrid">{items.map((item: Json, i: number) => <div className="equipmentItem" key={item.equipment_id || i}><div className="equipmentIcon"><Box size={18}/></div><div><b>{item.name || item.label || item.equipment_id}</b><small>{item.role || item.category || "process equipment"}</small><code>{item.equipment_id || "unassigned"}</code></div></div>)}</div> : <Empty icon={<Archive size={22}/>} title="No final allocation" text="Resolve inventory requirements before execution."/>}{unresolved.map((item: Json, i: number) => <div className="warningRow" key={i}><AlertTriangle size={16}/><span><b>{item.operation_id}</b>{item.reason}</span></div>)}</div>;
}


function ExperimentLoop({ job, result }: { job: Job | null; result: Json }) {
  const params = result.final_design?.parameters || {};
  const [experiments, setExperiments] = useState<Json[]>([]);
  const [form, setForm] = useState<Json>({ residence_time_min: params.residence_time_min || "", flow_rate_mL_min: params.flow_rate_mL_min || "", temperature_C: params.temperature_C || "", reactor_volume_mL: params.reactor_volume_mL || "", yield_pct: "", conversion_pct: "", notes: "" });
  const [refinement, setRefinement] = useState<Json | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const add = () => {
    if (form.yield_pct === "" && form.conversion_pct === "") return;
    const next = { run_id: `web_${experiments.length + 1}`, design_version: experiments.length + 1, actual_conditions: {
      residence_time_min: Number(form.residence_time_min), flow_rate_mL_min: Number(form.flow_rate_mL_min), temperature_C: Number(form.temperature_C), reactor_volume_mL: Number(form.reactor_volume_mL)
    }, outcomes: { yield_pct: form.yield_pct === "" ? null : Number(form.yield_pct), conversion_pct: form.conversion_pct === "" ? null : Number(form.conversion_pct), notes: form.notes }, free_text_observations: form.notes };
    setExperiments([...experiments, next]); setForm({...form, yield_pct: "", conversion_pct: "", notes: ""});
  };
  const refine = async () => {
    setBusy(true); setError("");
    try { setRefinement(await api<Json>("/api/refinement", { method: "POST", body: JSON.stringify({ job_id: job?.job_id === "demo" ? null : job?.job_id, current_result: job?.job_id === "demo" ? result : null, experiments }) })); }
    catch (e) { setError(String((e as Error).message || e)); } finally { setBusy(false); }
  };
  const hasGas = Boolean((result.final_design?.streams || []).some((s: Json) => s.phase === "gas"));
  const fields = [[hasGas ? "Residence time (inlet/STP)" : "Residence time", "residence_time_min", "min"], ["Liquid flow", "flow_rate_mL_min", "mL/min"], ["Temperature", "temperature_C", "°C"], ["Reactor volume", "reactor_volume_mL", "mL"], ["Yield", "yield_pct", "%"], ["Conversion", "conversion_pct", "%"]];
  return <div className="cycleLayout"><section><div className="sectionTitle"><h2>Add experimental result</h2><span>actual wet-lab values</span></div><div className="formGrid">{fields.map(([label, key, unit]) => <label className="field" key={key}><span>{label} <small>{unit}</small></span><input type="number" value={form[key]} onChange={(e) => setForm({...form, [key]: e.target.value})}/></label>)}</div><label className="field"><span>Observations</span><textarea value={form.notes} onChange={(e) => setForm({...form, notes: e.target.value})} placeholder="Pressure drift, precipitation, gas-liquid stability, impurities…"/></label><div className="actions"><button onClick={add}><Plus size={15}/>Add cycle</button><button className="primary" disabled={!experiments.length || busy} onClick={refine}><Sparkles size={15}/>Refine from {experiments.length} cycle{experiments.length === 1 ? "" : "s"}</button></div>{error && <div className="inlineError">{error}</div>}</section><section><div className="sectionTitle"><h2>Campaign history</h2><span>{experiments.length} cycles</span></div>{experiments.length ? <div className="cycleList">{experiments.map((e, i) => <div key={i}><span>v{i + 1}</span><div><b>{formatValue(e.outcomes.yield_pct ?? e.outcomes.conversion_pct)}%</b><small>{formatValue(e.actual_conditions.residence_time_min)} min · {formatValue(e.actual_conditions.flow_rate_mL_min)} mL/min</small></div><button onClick={() => setExperiments(experiments.filter((_, j) => i !== j))}><Trash2 size={14}/></button></div>)}</div> : <Empty icon={<TestTube2 size={22}/>} title="No wet-lab cycles" text="Add the actual conditions and analytical result from the first run."/>}{refinement && <div className="refinement"><span className="eyebrow">Next design</span><h3>{refinement.decision?.status?.replaceAll("_", " ")}</h3><p>{refinement.decision?.diagnosis}</p><TagList values={refinement.decision?.recommended_actions || []}/><JsonViewer value={refinement.decision?.next_experiment || {}} compact/></div>}</section></div>;
}


function InventorySnapshot({ profile }: { profile: InventoryProfile }) {
  const inv = profile.lab_inventory || {};
  return <div className="snapshot"><div><span>Reactors</span><b>{inv.reactors?.map((r: Json) => `${r.name || r.system} (${r.volume_mL} mL)`).join(", ") || "none"}</b></div><div><span>Pumps</span><b>{inv.pumps?.map((p: Json) => p.name || p.type).join(", ") || "none"}</b></div><div><span>Pressure</span><b>{inv.pressure_controllers?.map((p: Json) => p.name).join(", ") || inv.BPR_available?.join(", ") || "none"}</b></div><div><span>Constraints</span><b>{Object.keys(profile.operating_constraints || {}).length} declared</b></div></div>;
}

function equipmentCounts(inv: Json): [string, number][] {
  return [["reactors", inv.reactors?.length || 0], ["pumps", inv.pumps?.length || 0], ["tubing", inv.tubing?.length || 0], ["lights", inv.light_sources?.length || 0], ["gas devices", inv.gas_hardware?.length || 0], ["accessories", ["mixers","separators","connectors","collectors"].reduce((n, k) => n + (inv[k]?.length || 0), 0)]];
}

function RunsView({ runs, onOpen }: { runs: Json[]; onOpen: (runId: string) => void }) {
  return <section className="panel runsPanel"><div className="panelHead"><div><span className="eyebrow">Provenance</span><h2><History size={17}/>Autosaved designs</h2></div><span className="tag">{runs.length} recent runs</span></div>{runs.length ? <table className="dataTable"><thead><tr><th>Run</th><th>Reaction</th><th>Status</th><th>Residence</th><th>Flow</th><th>Reactor</th><th>Open</th></tr></thead><tbody>{runs.map((r) => <tr key={r.run_id}><td><b>{r.run_id}</b><small>{r.source}</small></td><td>{r.reaction || "--"}</td><td><span className={`statusPill ${r.final_design_status === "executable" ? "success" : "danger"}`}>{r.final_design_status}</span></td><td>{formatValue(r.residence_time_min)} min</td><td>{formatValue(r.flow_rate_mL_min)} mL/min</td><td>{formatValue(r.reactor_volume_mL)} mL</td><td><button className="openRun" onClick={() => onOpen(r.run_id)} title="Open saved design"><ExternalLink size={14}/></button></td></tr>)}</tbody></table> : <Empty icon={<History size={24}/>} title="No autosaved runs" text="Completed web and Streamlit designs appear here automatically."/>}</section>;
}

function SystemView({ health }: { health: Health | null }) {
  const models = health?.models || {};
  return <div className="systemGrid"><section className="panel"><div className="panelHead"><h2><Activity size={17}/>Core services</h2><span className="statusPill success">online</span></div><div className="systemRows"><div><span>API</span><b>FastAPI · local</b><CheckCircle2 size={16}/></div><div><span>Corpus</span><b>{health?.corpus_records || 0} records</b><Database size={16}/></div><div><span>Inventory</span><b>{health?.inventory_profiles || 0} profiles</b><Archive size={16}/></div></div></section><section className="panel"><div className="panelHead"><h2><Bot size={17}/>Model routing</h2></div><div className="modelRoute"><div><span>Intake</span><b>{models.intake || "--"}</b></div><ChevronDown size={16}/><div><span>Chemistry</span><b>{models.chemistry || "--"}</b></div><ChevronDown size={16}/><div><span>Translation</span><b>{models.translation || "--"}</b></div><ChevronDown size={16}/><div><span>Council</span><b>{models.council_model || "--"}</b></div></div></section><section className="panel fullSpan"><div className="panelHead"><h2><ShieldCheck size={17}/>Authority boundaries</h2></div><div className="authorityDiagram"><div><TestTube2 size={20}/><b>Measured evidence</b><span>Overrides model assumptions</span></div><ArrowRight size={18}/><div><Archive size={20}/><b>Hard constraints</b><span>Validated deterministically</span></div><ArrowRight size={18}/><div><FileText size={20}/><b>Protocol facts</b><span>Frozen during intake</span></div><ArrowRight size={18}/><div><Bot size={20}/><b>Model inference</b><span>Advisory until closure</span></div></div></section></div>;
}

function JsonViewer({ value, compact = false }: { value: any; compact?: boolean }) {
  const [copied, setCopied] = useState(false);
  const text = JSON.stringify(value, null, 2);
  const copy = async () => { await navigator.clipboard.writeText(text); setCopied(true); window.setTimeout(() => setCopied(false), 1200); };
  return <div className={`jsonViewer ${compact ? "compact" : ""}`}><button onClick={copy}>{copied ? <Check size={14}/> : <FileText size={14}/>} {copied ? "Copied" : "Copy"}</button><pre>{text}</pre></div>;
}

function TagList({ values }: { values: any[] }) { return <div className="tagList">{values.map((v, i) => <span key={i}>{typeof v === "string" ? v : JSON.stringify(v)}</span>)}</div>; }
function Empty({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) { return <div className="emptyState">{icon}<b>{title}</b><span>{text}</span></div>; }

createRoot(document.getElementById("root")!).render(<App/>);
