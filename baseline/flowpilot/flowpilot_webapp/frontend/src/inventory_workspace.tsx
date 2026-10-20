import React, { useEffect, useRef, useState } from "react";
import { Archive, ArrowRight, CheckCircle2, Download, FileJson, ListChecks, Pencil, Plus, Save, Settings2, ShieldCheck, Trash2, Upload, X } from "lucide-react";

type Json = Record<string, any>;
type Profile = Json & { profile_id: string; name: string; lab_inventory: Json };
type Category = { key: string; label: string; schema: Json };
type Metadata = { categories: Category[]; inventory_schema: Json; empty_profile: Profile };
type Api = <T>(url: string, init?: RequestInit) => Promise<T>;
const STORAGE = "flowpilot_inventory_editor_v1";
const clone = <T,>(value: T): T => structuredClone(value);
const newProfileId = () => `manual_${Array.from(crypto.getRandomValues(new Uint8Array(12)), n => n.toString(16).padStart(2, "0")).join("")}`;
const human = (key: string) => key.replaceAll("_", " ").replace(/^./, c => c.toUpperCase());
const LABELS: Record<string, string> = {
  name: "Equipment name", type: "Type", equipment_id: "Equipment ID", quantity: "Quantity",
  min_flow_rate_mL_min: "Minimum flow (mL/min)", max_flow_rate_mL_min: "Maximum flow (mL/min)",
  flow_rate_increment_mL_min: "Flow setting increment (mL/min)", max_pressure_bar: "Maximum pressure (bar)",
  min_pressure_bar: "Minimum pressure (bar)", ID_mm: "Internal diameter (mm)", volume_mL: "Volume (mL)",
  min_temperature_C: "Minimum temperature (C)", max_temperature_C: "Maximum temperature (C)",
  allowed_temperatures_C: "Discrete temperatures (C)", wavelength_nm: "Wavelength (nm)", power_W: "Power (W)",
  min_concentration_M: "Minimum concentration (M)", max_concentration_M: "Maximum concentration (M)",
  intensity_mW_cm2: "Intensity (mW/cm2)", min_flow_sccm: "Minimum gas flow (mL/min, STP)",
  max_flow_sccm: "Maximum gas flow (mL/min, STP)", flow_rate_increment_sccm: "Gas flow increment (mL/min, STP)",
  compatible_systems: "Compatible systems", platform_id: "Pump platform ID", photoreactor_module_ids: "Compatible light module IDs",
};
const BASIC = ["name", "type", "quantity", "material", "volume_mL", "ID_mm", "system", "platform_id",
  "min_flow_rate_mL_min", "max_flow_rate_mL_min", "flow_rate_increment_mL_min", "max_pressure_bar",
  "min_temperature_C", "max_temperature_C", "wavelength_nm", "power_W", "compatible_reactor",
  "gas", "min_flow_sccm", "max_flow_sccm", "compatible_systems", "photoreactor_module_ids", "notes"];

function fieldType(field: Json): Json {
  return field.anyOf?.find((option: Json) => option.type !== "null") || field;
}

function Field({ id, spec, required, value, set }: { id: string; spec: Json; required: boolean; value: string; set: (value: string) => void }) {
  const type = fieldType(spec);
  const label = LABELS[id] || human(id);
  const props = { "aria-label": label, value, onChange: (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => set(event.target.value), required };
  return <label className={`field inventoryField ${["array", "object"].includes(type.type) || id === "notes" ? "wideField" : ""}`}>
    <span>{label}{required && <span aria-hidden="true"> *</span>}{type.type === "array" && <small>one value per line</small>}{type.type === "object" && <small>JSON object</small>}</span>
    {type.type === "boolean" ? <select {...props}><option value="">Not specified</option><option value="true">Yes</option><option value="false">No</option></select>
      : type.enum ? <select {...props}><option value="">Select</option>{type.enum.map((v: string) => <option key={v}>{v}</option>)}</select>
      : ["object", "array"].includes(type.type) || id === "notes" ? <textarea {...props} rows={3} spellCheck={false}/>
      : <input {...props} type={["number", "integer"].includes(type.type) ? "number" : "text"} step={type.type === "integer" ? 1 : "any"} min={type.minimum} max={type.maximum}/>}</label>;
}

function EquipmentForm({ category, item, onCommit, onClose }: { category: Category; item: Json; onCommit: (item: Json) => void; onClose: () => void }) {
  const properties = category.schema.properties;
  const [values, setValues] = useState<Record<string, string>>(() => Object.fromEntries(Object.entries(properties).map(([key, spec]: [string, any]) => {
    const value = item[key] ?? spec.default;
    return [key, value == null ? "" : Array.isArray(value) ? value.join("\n") : typeof value === "object" ? JSON.stringify(value, null, 2) : String(value)];
  })));
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  const required: string[] = category.schema.required || [];
  const keys = Object.keys(properties);
  const basic = [...new Set([...BASIC.filter(k => keys.includes(k)), ...required])];
  const advanced = keys.filter(k => !basic.includes(k));
  const field = (id: string) => <Field key={id} id={id} spec={properties[id]} required={required.includes(id)} value={values[id] || ""} set={v => setValues(old => ({ ...old, [id]: v }))}/>;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const next = { ...item };
      for (const [key, spec] of Object.entries(properties) as [string, Json][]) {
        const value = values[key]?.trim() || "";
        const type = fieldType(spec);
        if (!value) {
          if (required.includes(key)) throw new Error(`${LABELS[key] || human(key)} is required.`);
          if (spec.anyOf?.some((v: Json) => v.type === "null")) next[key] = null;
          else if (type.type === "array") next[key] = [];
          else if (type.type === "object") next[key] = {};
          else if ("default" in spec) next[key] = spec.default;
          else delete next[key];
        } else if (["number", "integer"].includes(type.type)) {
          const n = Number(value);
          if (!Number.isFinite(n) || (type.type === "integer" && !Number.isInteger(n))) throw new Error(`${human(key)} must be a finite ${type.type}.`);
          next[key] = n;
        } else if (type.type === "boolean") next[key] = value === "true";
        else if (type.type === "array") {
          const numeric = ["number", "integer"].includes(type.items?.type);
          next[key] = value.split(numeric ? /[\n,]+/ : /\n+/).map(v => v.trim()).filter(Boolean).map(v => {
            if (!numeric) return v;
            const n = Number(v);
            if (!Number.isFinite(n)) throw new Error(`${human(key)} contains an invalid number: ${v}`);
            return n;
          });
        } else if (type.type === "object") {
          next[key] = JSON.parse(value);
          if (!next[key] || typeof next[key] !== "object" || Array.isArray(next[key])) throw new Error(`${human(key)} must be a JSON object.`);
        } else next[key] = value;
      }
      onCommit(next);
    } catch (e) { setError((e as Error).message); }
  };
  return <dialog ref={dialog} className="equipmentDialog" onCancel={onClose} onClose={onClose}>
    <form onSubmit={submit}><header><div><span className="eyebrow">{category.label}</span><h2>{item.equipment_id ? "Edit equipment" : "Add equipment"}</h2></div><button type="button" title="Close equipment form" onClick={onClose}><X size={19}/></button></header>
      <div className="equipmentFormBody"><div className="equipmentFields">{basic.map(field)}</div><details className="inventoryAdvanced"><summary>Identification, compatibility and additional limits</summary><div className="equipmentFields">{advanced.map(field)}</div></details>{error && <div role="alert" className="inlineError">{error}</div>}</div>
      <footer><button type="button" onClick={onClose}>Cancel</button><button type="submit" className="primary"><CheckCircle2 size={16}/>Apply equipment</button></footer>
    </form></dialog>;
}

function ObjectEditor({ label, value, commit, pending }: { label: string; value: Json; commit: (value: Json) => void; pending: (value: boolean) => void }) {
  const [text, setText] = useState(JSON.stringify(value, null, 2));
  const [error, setError] = useState("");
  useEffect(() => { setText(JSON.stringify(value, null, 2)); setError(""); }, [value]);
  return <div className="constraintObject"><label className="field"><span>{label} (JSON)</span><textarea value={text} onChange={e => { setText(e.target.value); pending(true); }} spellCheck={false}/></label>
    <button onClick={() => { try { const v = JSON.parse(text); if (!v || Array.isArray(v) || typeof v !== "object") throw new Error("Enter a JSON object."); commit(v); pending(false); setError(""); } catch (e) { setError((e as Error).message); } }}><CheckCircle2 size={14}/>Apply {label.toLowerCase()}</button>{error && <p role="alert" className="inlineError">{error}</p>}</div>;
}

function restore() {
  try { return JSON.parse(sessionStorage.getItem(STORAGE) || "{}"); } catch { return {}; }
}

export function InventoryWorkspace({ profiles, refresh, onUse, api }: { profiles: Json[]; refresh: () => void; onUse: (profile: Profile) => void; api: Api }) {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [draft, setDraft] = useState<Profile | null>(() => restore().draft || null);
  const [tab, setTab] = useState<string>(() => restore().tab || "Equipment");
  const [categoryKey, setCategory] = useState<string>(() => restore().category || "pumps");
  const [editing, setEditing] = useState<{ index: number; item: Json } | null>(null);
  const [dirty, setDirty] = useState(true);
  const [jsonText, setJsonText] = useState(() => restore().jsonText || "");
  const [rawDirty, setRawDirty] = useState(() => Boolean(restore().rawDirty));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [pendingObjects, setPendingObjects] = useState<Record<string, boolean>>({});
  const pending = Object.values(pendingObjects).some(Boolean);
  const [sourceText, setSourceText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [useLlm, setUseLlm] = useState(true);
  const fileInput = useRef<HTMLInputElement>(null);
  const jsonInput = useRef<HTMLInputElement>(null);
  const initialized = useRef(false);
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    api<Metadata>("/api/inventory/editor-schema").then(m => { setMetadata(m); setDraft(current => current || { ...m.empty_profile, profile_id: newProfileId() }); }).catch(e => setError(e.message));
  }, []);
  useEffect(() => { try { sessionStorage.setItem(STORAGE, JSON.stringify({ draft, tab, category: categoryKey, jsonText, rawDirty })); } catch { /* Browser storage is optional; export remains available. */ } }, [draft, tab, categoryKey, jsonText, rawDirty]);
  const change = (next: Profile) => { setDraft({ ...next, status: "draft" }); setDirty(true); setError(""); setNotice(""); };
  const accept = (next: Profile) => { setDraft(next); setJsonText(JSON.stringify(next, null, 2)); setRawDirty(false); setDirty(false); setPendingObjects({}); };
  const task = async (fn: () => Promise<void>) => { setBusy(true); setError(""); setNotice(""); try { await fn(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } };
  const switchTab = (next: string) => {
    if (pending) { setError("Apply the additional constraint edits before changing sections."); return; }
    if (rawDirty && next !== "JSON") { setError("Validate the JSON edits before returning to the forms, or discard the JSON edits."); return; }
    setTab(next); if (next === "JSON" && !rawDirty) setJsonText(JSON.stringify(draft, null, 2)); setError("");
  };
  const validate = () => task(async () => {
    if (pending) throw new Error("Apply the additional constraint edits before validation.");
    const payload = rawDirty ? JSON.parse(jsonText) : clone(draft);
    if (!payload?.name?.trim()) throw new Error("A profile name is required.");
    if (!rawDirty) {
      const alternatives = payload.equipment_capabilities?.inline_degassing?.allowed_alternatives;
      if (Array.isArray(alternatives)) payload.equipment_capabilities.inline_degassing.allowed_alternatives = alternatives.map((s: string) => s.trim()).filter(Boolean);
      const forbidden = payload.operating_constraints?.forbidden_equipment;
      if (Array.isArray(forbidden)) payload.operating_constraints.forbidden_equipment = forbidden.map((s: string) => s.trim()).filter(Boolean);
    }
    const next = await api<Profile>("/api/inventory/import", { method: "POST", body: JSON.stringify(payload) });
    accept(next); setNotice(next.validation?.valid ? "Profile validated. Ready to export, save or use in design." : "Review the validation findings before exporting or using this profile.");
  });
  const valid = Boolean(draft?.validation?.valid && !dirty && !rawDirty && !pending);
  const category = metadata?.categories.find(c => c.key === categoryKey);
  const equipment: Json[] = draft?.lab_inventory?.[categoryKey] || [];
  const importJson = (file: File) => task(async () => { const next = await api<Profile>("/api/inventory/import", { method: "POST", body: JSON.stringify(JSON.parse(await file.text())) }); accept(next); setTab("Equipment"); });
  const exportJson = () => {
    if (!valid || !draft) return;
    const blob = new Blob([JSON.stringify(draft, null, 2) + "\n"], { type: "application/json" });
    const href = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = href; link.download = `${draft.profile_id}_v${draft.version || 0}.json`; link.click(); window.setTimeout(() => URL.revokeObjectURL(href), 1000);
  };
  const load = (id: string) => task(async () => { if ((dirty || rawDirty || pending) && draft && !window.confirm("Replace the current inventory draft? Unsaved edits will be lost.")) return; accept(await api<Profile>(`/api/inventory/profiles/${encodeURIComponent(id)}`)); setTab("Equipment"); });
  const capabilities = draft?.equipment_capabilities || {};
  const inline = capabilities.inline_degassing;
  const limits = draft?.operating_constraints || {};
  const setInline = (value: string) => {
    if (!draft) return;
    const caps = clone(capabilities); const operating = clone(limits);
    delete operating.inline_degasser_available;
    // Keep only unrelated exclusions when changing the explicit degasser choice.
    const related = new Set(["inline_degasser", "inline_degassing", "membrane_degasser"]);
    if (Array.isArray(operating.forbidden_equipment)) operating.forbidden_equipment = operating.forbidden_equipment.filter((v: string) => !related.has(v));
    if (!value) delete caps.inline_degassing;
    else caps.inline_degassing = { ...inline, available: value === "available", service_status: value, allowed_alternatives: inline?.allowed_alternatives || [], notes: inline?.notes || "" };
    change({ ...draft, equipment_capabilities: caps, operating_constraints: operating });
  };
  const description = (item: Json) => [item.type || item.material, item.volume_mL != null ? `${item.volume_mL} mL` : null, item.min_flow_rate_mL_min != null ? `${item.min_flow_rate_mL_min}-${item.max_flow_rate_mL_min} mL/min` : null, item.wavelength_nm ? `${item.wavelength_nm} nm` : null, item.system || item.platform_id].filter(Boolean).join(" | ");
  return <div className="viewStack inventoryWorkspace" aria-busy={busy}>
    <header className="inventoryTitle"><div><span className="eyebrow">Laboratory inventory</span><h2>Equipment and constraints</h2></div><div className="actions"><button disabled={!metadata || busy} onClick={() => { if ((dirty || rawDirty || pending) && draft && !window.confirm("Create a new profile? Unsaved edits will be lost.")) return; const p = clone(metadata!.empty_profile); p.profile_id = newProfileId(); change(p); setPendingObjects({}); setRawDirty(false); setTab("Equipment"); }}><Plus size={16}/>New profile</button><button disabled={busy} onClick={() => jsonInput.current?.click()}><Upload size={16}/>Import JSON</button></div></header>
    <input ref={jsonInput} type="file" accept=".json,application/json" hidden aria-label="Import inventory JSON" onChange={e => { const file = e.target.files?.[0]; if (file) { if ((dirty || rawDirty) && draft && !window.confirm("Replace the current inventory draft with this JSON?")) { e.target.value = ""; return; } void importJson(file); } e.target.value = ""; }}/>
    {error && <div role="alert" className="inlineError">{error}</div>}
    {draft && metadata ? <fieldset className="inventoryEditorFieldset" disabled={busy}>
      <div className="inventoryIdentity"><label className="field"><span>Profile name</span><input disabled={rawDirty} value={draft.name} onChange={e => change({ ...draft, name: e.target.value })}/></label><label className="field"><span>Laboratory</span><input disabled={rawDirty} value={draft.laboratory || ""} onChange={e => change({ ...draft, laboratory: e.target.value })}/></label><span className={`statusPill ${valid ? "success" : "warning"}`}>{valid ? "Validated" : dirty || rawDirty ? "Unvalidated edits" : "Needs review"}</span></div>
      <nav className="inventoryTabs" aria-label="Inventory editor sections">{["Equipment", "Constraints", "Import documents", "JSON"].map(t => <button key={t} aria-pressed={tab === t} onClick={() => switchTab(t)}>{t === "Equipment" ? <Archive size={15}/> : t === "Constraints" ? <ShieldCheck size={15}/> : t === "JSON" ? <FileJson size={15}/> : <Upload size={15}/>} {t}</button>)}</nav>
      {tab === "Equipment" && category && <div className="equipmentLayout"><nav className="equipmentCategories" aria-label="Equipment categories">{metadata.categories.map(c => <button key={c.key} aria-pressed={categoryKey === c.key} onClick={() => setCategory(c.key)}><span>{c.label}</span><b>{draft.lab_inventory[c.key]?.length || 0}</b></button>)}</nav><section className="equipmentList"><header><h3>{category.label}</h3><button className="primary" onClick={() => setEditing({ index: -1, item: {} })}><Plus size={16}/>Add equipment</button></header><label className="categoryStatus"><span>Availability</span><select aria-label={`${category.label} availability`} value={draft.lab_inventory.capability_status?.[categoryKey] || "undocumented"} onChange={e => change({ ...draft, lab_inventory: { ...draft.lab_inventory, capability_status: { ...draft.lab_inventory.capability_status, [categoryKey]: e.target.value } } })}><option value="undocumented">Not confirmed</option><option value="available">Available</option><option value="unavailable">Unavailable</option></select></label>
        {!equipment.length && <div className="inventoryEmpty">No {category.label.toLowerCase()} entered.</div>}
        <div className="equipmentRows">{equipment.map((item, index) => <div className="equipmentRow" key={`${categoryKey}-${index}`}><div><strong>{item.name || item.equipment_id || `${category.label} ${index + 1}`}</strong><p>{description(item)}</p><small>Quantity {item.quantity ?? 1} | {item.service_status || "available"} | {item.equipment_id || "ID assigned on validation"}</small></div><div className="equipmentRowActions"><button title={`Edit ${item.name || category.label}`} onClick={() => setEditing({ index, item })}><Pencil size={16}/></button><button title={`Delete ${item.name || category.label}`} onClick={() => { if (window.confirm(`Remove ${item.name || category.label} from this draft?`)) change({ ...draft, lab_inventory: { ...draft.lab_inventory, [categoryKey]: equipment.filter((_, i) => i !== index) } }); }}><Trash2 size={16}/></button></div></div>)}</div></section></div>}
      {tab === "Constraints" && <section className="inventoryConstraints"><h3>Operating constraints</h3><div className="equipmentFields"><label className="field"><span>Inline degasser</span><select value={inline ? inline.available ? "available" : "unavailable" : ""} onChange={e => setInline(e.target.value)}><option value="">Not specified</option><option value="available">Available</option><option value="unavailable">Unavailable</option></select></label><label className="field"><span>Standard reactor connectors available</span><select value={String(draft.lab_inventory.standard_reactor_connectors_available)} onChange={e => change({ ...draft, lab_inventory: { ...draft.lab_inventory, standard_reactor_connectors_available: e.target.value === "true" } })}><option value="true">Yes</option><option value="false">No</option></select></label>
      <label className="field"><span>Allowed degassing alternatives<small>one per line</small></span><textarea disabled={!inline} value={(inline?.allowed_alternatives || []).join("\n")} onChange={e => change({ ...draft, equipment_capabilities: { ...capabilities, inline_degassing: { ...inline, allowed_alternatives: e.target.value.split("\n") } } })}/></label><label className="field"><span>Forbidden equipment<small>one per line</small></span><textarea value={(limits.forbidden_equipment || []).join("\n")} onChange={e => change({ ...draft, operating_constraints: { ...limits, forbidden_equipment: e.target.value.split("\n") } })}/></label></div>
      <details className="inventoryAdvanced"><summary><Settings2 size={15}/> Additional limits, capabilities and shared resources</summary><ObjectEditor pending={v => setPendingObjects(old => ({ ...old, "Operating constraints": v }))} label="Operating constraints" value={limits} commit={v => change({ ...draft, operating_constraints: v })}/><ObjectEditor pending={v => setPendingObjects(old => ({ ...old, "Equipment capabilities": v }))} label="Equipment capabilities" value={capabilities} commit={v => change({ ...draft, equipment_capabilities: v })}/><ObjectEditor pending={v => setPendingObjects(old => ({ ...old, "Resource capacities": v }))} label="Resource capacities" value={draft.lab_inventory.resource_capacities || {}} commit={v => change({ ...draft, lab_inventory: { ...draft.lab_inventory, resource_capacities: v } })}/></details></section>}
      {tab === "Import documents" && <section className="inventorySource"><h3>Import laboratory inventory</h3><button className="dropZone" onClick={() => fileInput.current?.click()}><Upload size={24}/><b>Choose PDF, Word, PowerPoint, spreadsheet, or JSON</b><span>{files.length ? files.map(f => f.name).join(", ") : "Multiple source documents"}</span></button><input ref={fileInput} type="file" hidden multiple accept=".pdf,.docx,.pptx,.xlsx,.xlsm,.csv,.tsv,.txt,.md,.json" onChange={e => setFiles(Array.from(e.target.files || []))}/><label className="field"><span>Additional equipment and constraints</span><textarea value={sourceText} onChange={e => setSourceText(e.target.value)}/></label><div className="inlineControls"><label className="toggle"><input type="checkbox" checked={useLlm} onChange={e => setUseLlm(e.target.checked)}/><i/><span>LLM-assisted extraction</span></label><button className="primary" disabled={!files.length && !sourceText.trim()} onClick={() => task(async () => {
        if (!window.confirm("Replace the equipment draft with the extracted inventory? Review extraction before saving.")) return;
        if (files.some(f => f.name.toLowerCase().endsWith(".json"))) {
          if (files.length !== 1 || sourceText.trim()) throw new Error("Import a JSON profile on its own, then add constraints in the Constraints section.");
          accept(await api<Profile>("/api/inventory/import", { method: "POST", body: JSON.stringify(JSON.parse(await files[0].text())) })); setTab("Equipment"); return;
        }
        const body = new FormData(); body.append("name", draft.name); body.append("laboratory", draft.laboratory || ""); body.append("source_text", sourceText); body.append("use_llm", String(useLlm)); files.forEach(f => body.append("files", f));
        const p = await api<Profile>("/api/inventory/extract", { method: "POST", body }); accept(p); setTab("Equipment"); if (p.document_errors?.length) setError(p.document_errors.join("\n"));
      })}><Upload size={15}/>Extract inventory</button></div></section>}
      {tab === "JSON" && <section className="inventoryRaw"><label className="field"><span>Inventory profile JSON</span><textarea value={jsonText} onChange={e => { setJsonText(e.target.value); setRawDirty(true); setDirty(true); }} spellCheck={false}/></label>{rawDirty && <button onClick={() => { setJsonText(JSON.stringify(draft, null, 2)); setRawDirty(false); }}>Discard JSON edits</button>}</section>}
      <section className="inventoryValidation" aria-label="Inventory validation"><div><h3><ListChecks size={17}/>Validation and export</h3><p role="status">{notice || (valid ? "Profile validated." : "Validation required before export or design.")}</p></div>
        {!dirty && !rawDirty && <div className="validationList">{draft.validation?.errors?.map((v: string) => <div key={v} className="bad">{v}</div>)}{draft.validation?.unresolved_fields?.map((v: string) => <div key={v} className="bad">Unresolved: {v}</div>)}{Boolean(draft.validation?.warnings?.length) && <details><summary>{draft.validation.warnings.length} validation warnings</summary>{draft.validation.warnings.map((v: string) => <p key={v}>{v}</p>)}</details>}</div>}
        <div className="actions"><button onClick={validate}><CheckCircle2 size={16}/>Validate profile</button><button disabled={!valid} onClick={exportJson}><Download size={16}/>Export JSON</button><button disabled={!valid} onClick={() => task(async () => { const r = await api<{ profile: Profile }>("/api/inventory/save", { method: "POST", body: JSON.stringify({ profile: draft }) }); accept(r.profile); refresh(); setNotice(`Saved version ${r.profile.version}.`); })}><Save size={16}/>Save profile</button><button className="primary" disabled={!valid} onClick={() => onUse(draft)}><ArrowRight size={16}/>Use in design</button></div></section>
    </fieldset> : <p>Loading inventory fields...</p>}
    <section className="savedProfiles"><div className="panelHead"><h2><Archive size={17}/>Saved profiles</h2><span className="tag">{profiles.length} profiles</span></div><div className="profileRows">{profiles.map(p => <div key={p.profile_id}><div><b>{p.name}</b><small>{p.laboratory || "Laboratory not specified"} | version {p.version}</small></div><span>{p.reactor_count} reactors | {p.pump_count} pumps</span><button disabled={busy} onClick={() => load(p.profile_id)}><Pencil size={15}/>Open profile</button></div>)}</div></section>
    {editing && category && draft && <EquipmentForm category={category} item={editing.item} onClose={() => setEditing(null)} onCommit={item => { const rows = [...equipment]; if (editing.index < 0) rows.push(item); else rows[editing.index] = item; change({ ...draft, lab_inventory: { ...draft.lab_inventory, [categoryKey]: rows } }); setEditing(null); }}/ >}
  </div>;
}
