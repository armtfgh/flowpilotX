import React from "react";
import { AlertTriangle, Download } from "lucide-react";

type Json = Record<string, any>;
const num = (value: any) => value == null ? "--" : typeof value === "number" ? Number(value.toPrecision(6)).toString() : String(value);
const asText = (value: any) => value == null ? "Not recorded" : typeof value === "string" ? value : JSON.stringify(value, null, 2);
// This is a unit spelling change only; archived transcript values stay intact.
export const displayUnits = (text: string) => text.replace(/\bsccm\b/gi, "mL/min at STP");

function RecordText({value}: {value: any}) {
  return <pre className="recordText">{asText(value)}</pre>;
}

export function StageTable({rows}: {rows: Json[]}) {
  return <div className="tableScroll"><table className="dataTable processTable"><thead><tr>
    <th>Stage / reactor</th><th>Volume<br/>mL</th><th>Liquid<br/>mL/min</th><th>Gas at STP<br/>mL/min</th><th>Time<br/>min</th><th>Temperature<br/>°C</th><th>Pressure<br/>bar</th><th>Light<br/>nm</th>
  </tr></thead><tbody>{rows.map((s, i) => <tr key={i} data-stage={s.number}>
    <td><b>Stage {s.number}: {s.name}</b><span>{s.reactor}</span><small>{s.material} · {num(s.tubing_ID_mm)} mm ID</small></td>
    <td>{num(s.volume_mL)}</td><td>{num(s.liquid_flow_mL_min)}</td><td>{num(s.gas_flow_stp_mL_min)}</td>
    <td><b>{num(s.residence_time_min)}</b><small>{s.residence_basis}</small></td><td>{num(s.temperature_C)}</td>
    <td>{num(s.pressure_bar)}<small>{s.pressure_basis}</small></td><td>{num(s.wavelength_nm)}</td>
  </tr>)}</tbody></table></div>;
}

export function ProcessSummary({result}: {result: Json}) {
  const report = result.result_report || {};
  const stages = report.stages || [];
  const streams = report.streams || [];
  const download = () => {
    const rows = [
      ["record", "stage_or_stream", "name", "phase", "volume_mL", "liquid_flow_mL_min", "gas_flow_STP_mL_min", "residence_inlet_min", "temperature_C", "equiv", "equipment"],
      ...stages.map((s: Json) => ["reactor", s.number, s.name, "", s.volume_mL, s.liquid_flow_mL_min, s.gas_flow_stp_mL_min, s.residence_time_min, s.temperature_C, "", s.reactor]),
      ...streams.map((s: Json) => ["stream", s.label, (s.contents || []).join("; "), s.phase, "", s.phase === "gas" ? "" : s.flow_mL_min, s.phase === "gas" ? s.flow_mL_min : "", "", "", s.equiv, s.equipment_name]),
    ];
    const csv = rows.map(row => row.map((v: any) => `"${String(v ?? "").replaceAll('"', '""')}"`).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
    const a = document.createElement("a"); a.href = url; a.download = "flowpilot_process_summary.csv"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  if (report.status !== "executable") return <p>No accepted process setpoints are available.</p>;
  return <div className="reportStack">
    <div className="sectionTitle"><h2>Process summary</h2><button className="secondary" onClick={download}><Download size={14}/>CSV</button></div>
    <p className="reportNote">Gas reference: 273.15 K, 1.01325 bar. Gas-stage time = reactor volume / (liquid flow + inlet/STP gas flow); liquid-stage time = volume / liquid flow.</p>
    {(report.issues || []).map((s: string) => <div className="warningRow" key={s}><AlertTriangle size={16}/>{s}</div>)}
    <StageTable rows={stages}/>
    <h3>Feed streams</h3>
    <div className="tableScroll"><table className="dataTable streamTable"><thead><tr><th>Stream / enters</th><th>Composition</th><th>Phase</th><th>Flow<br/>mL/min</th><th>Feed concentration<br/>M</th><th>Reagent equiv</th><th>Delivery equipment</th></tr></thead>
      <tbody>{streams.map((s: Json, i: number) => <tr key={i}><td><b>{s.label}</b>Stage {s.introduction_stage ?? "not recorded"}</td>
        <td>{(s.contents || []).join("; ")}</td><td>{s.phase}</td><td><b>{num(s.flow_mL_min)}</b><small>{s.flow_basis}</small></td><td>{s.phase === "gas" ? "Not applicable" : num(s.concentration_M)}</td><td>{num(s.equiv)}{s.phase === "gas" && <small>inlet/STP; active gas fraction {num(s.reagent_fraction)}</small>}</td><td>{s.equipment_name || "Not recorded"}</td></tr>)}</tbody></table></div>
  </div>;
}

export function StageOverview({result, closure}: {result: Json; closure: React.ReactNode}) {
  const report = result.result_report || {};
  return <div className="reportStack">
    <div className="sectionTitle"><h2>Stage-by-stage design</h2><span>{report.stages?.length || 0} reactors</span></div>
    <div className="overviewStagesTable"><StageTable rows={report.stages || []}/></div>
    <div className="stageOverviewCompact">{(report.stages || []).map((s: Json) => <section key={s.number} data-stage={s.number}>
      <h3>Stage {s.number}: {s.name}</h3><p>{s.reactor}</p>
      <dl className="metricRecord"><dt>Time ({s.residence_basis})</dt><dd>{num(s.residence_time_min)} min</dd>
        <dt>Liquid flow</dt><dd>{num(s.liquid_flow_mL_min)} mL/min</dd><dt>Gas flow at STP</dt><dd>{num(s.gas_flow_stp_mL_min)} mL/min</dd>
        <dt>Temperature</dt><dd>{num(s.temperature_C)} °C</dd><dt>Volume</dt><dd>{num(s.volume_mL)} mL</dd>
        <dt>Pressure ({s.pressure_basis})</dt><dd>{num(s.pressure_bar)} bar</dd><dt>Wavelength</dt><dd>{num(s.wavelength_nm)} nm</dd></dl>
    </section>)}</div>
    {(report.issues || []).map((s: string) => <div className="warningRow" key={s}>{s}</div>)}
    <section><div className="sectionTitle"><h2>Closure</h2></div>{closure}</section>
  </div>;
}

const metricFields = [
  ["reactor_volume_mL", "Reactor volume (mL)"], ["flow_rate_mL_min", "Liquid flow (mL/min)"],
  ["residence_time_min", "Residence time (min)"], ["temperature_C", "Temperature (°C)"],
  ["tubing_ID_mm", "Tubing ID (mm)"], ["tubing_length_m", "Tubing length (m)"],
  ["reynolds_number", "Reynolds number"], ["pressure_drop_bar", "Estimated pressure drop (bar)"],
  ["UA_W_K", "Estimated UA (W/K)"], ["heat_generation_W", "Estimated heat release (W)"],
  ["Pe", "Estimated Peclet number"], ["intensification_factor", "Implied batch/flow time ratio"],
];

function MetricRecord({value}: {value: Json}) {
  return <dl className="metricRecord">{metricFields.filter(([key]) => value[key] != null).map(([key, label]) => <React.Fragment key={key}><dt>{label}</dt><dd>{num(value[key])}</dd></React.Fragment>)}</dl>;
}

function CalculatorRecord({value}: {value: Json}) {
  return <><MetricRecord value={value}/>
    {(value.steps || []).map((step: Json, i: number) => <details className="calculatorStep" key={i}>
      <summary>Step {step.step ?? i + 1}: {step.name} · {step.status}</summary>
      <RecordText value={step}/>
    </details>)}
  </>;
}

function ProposalSnapshot({value}: {value: Json}) {
  const stages = value.stage_parameters || [];
  if (!Object.keys(value).length) return <p>Not stored in this run.</p>;
  return <><p className="reportNote">Historical proposal, not run instructions. Recorded time basis: {value.residence_time_basis || "unspecified"}.</p>
    {stages.length ? <div className="tableScroll"><table className="dataTable snapshotTable"><thead><tr><th>Stage</th><th>Volume (mL)</th><th>Time (min, as proposed)</th><th>Liquid flow (mL/min)</th><th>Temperature (°C)</th></tr></thead><tbody>{stages.map((s: Json, i: number) => <tr key={i}><td>{s.stage_number || i + 1}</td><td>{num(s.reactor_volume_mL ?? s.V_R_mL)}</td><td>{num(s.residence_time_min)}</td><td>{num(s.Q_liquid_mL_min ?? s.flow_rate_mL_min)}</td><td>{num(s.temperature_C)}</td></tr>)}</tbody></table></div> : <MetricRecord value={value}/>}</>;
}

export function EngineeringHistory({result}: {result: Json}) {
  const history = result.engineering_history || {};
  const before = history.before_council || {};
  const after = history.after_council || {};
  const finalRecords = result.final_stage_engineering?.stages || [];
  return <div className="reportStack engineeringHistory">
    <ScientificAssessment result={result}/>
    <div className="sectionTitle"><h2>Engineering provenance</h2><span>proposal → council → final realization</span></div>
    <details className="auditSection" open><summary>1. Before council</summary><div className="auditBody">
      <ProposalSnapshot value={before.proposal || result.pre_council_proposal || {}}/>
      <details><summary>Initial engineering calculator</summary>{before.calculations ? <CalculatorRecord value={before.calculations}/> : <p>The initial calculator snapshot was not saved in this older run. It has not been reconstructed as historical evidence.</p>}</details>
    </div></details>
    <details className="auditSection"><summary>2. Council-selected candidate</summary><div className="auditBody"><ProposalSnapshot value={after.proposal || {}}/>
      <p className="reportNote">Selection precedes inventory realization and final validation.</p>
      {after.calculations && <CalculatorRecord value={after.calculations}/>}</div></details>
    <section className="finalEngineering"><div className="sectionTitle"><h2>3. Final design after council and inventory realization</h2></div>
      <StageTable rows={result.result_report?.stages || []}/>
      <p className="reportNote">Final setpoints are frozen. Engineering annotations below are recalculated for each realized reactor, not for one aggregate coil.</p>
      {!finalRecords.length && <p>Per-reactor engineering annotations were not saved in this run. The final stage setpoints above remain the authoritative design.</p>}
      {finalRecords.map((r: Json, i: number) => <details className="auditSection" key={i}><summary>Stage {r.stage_number} engineering: {r.status}</summary><div className="auditBody">
        {r.error ? <div className="warningRow">{r.error}</div> : <><MetricRecord value={r.calculations || {}}/>
          <p className="reportNote">Transport and thermal metrics are engineering estimates; they do not establish conversion or laboratory safety.</p></>}
      </div></details>)}
    </section>
    <details className="auditSection"><summary>Recorded realization decisions</summary><div className="auditBody"><RecordText value={result.design_realization?.decisions}/></div></details>
  </div>;
}

export function ScientificAssessment({result}: {result: Json}) {
  const a = result.scientific_assessment || result.proposal?.scientific_design;
  if (!a?.candidate_count) return null;
  const selected = (a.candidates || []).find((c: Json) => c.candidate_id === a.selected_candidate_id);
  return <section className="reportStack scientificAssessment" aria-label="Scientific assessment">
    <div className="sectionTitle"><h2>Scientific assessment</h2><span>Private 2.0 preview</span></div>
    {result.verification_provenance?.synthetic_objective_probe && <div className="warningRow"><AlertTriangle size={18}/><span>Synthetic objective-sensitivity test, not a user-requested experimental recommendation.</span></div>}
    <div className="warningRow"><AlertTriangle size={18}/><span>Experimental screen, not a validated high-yield process. Flow kinetics and achieved yield are unknown.</span></div>
    <dl className="metricRecord"><dt>Complete candidates</dt><dd>{a.candidate_count}</dd><dt>Selected experiment</dt><dd>{a.selected_candidate_id}</dd><dt>Selection preserved</dt><dd>{a.selected_design_preserved === true ? "Yes" : "Not verified"}</dd></dl>
    <h3>Council assessment · unvalidated</h3><p>{a.chief?.justification}</p>
    {a.objective_policy && <div className="objectiveAssessment"><h3>Objective and selection</h3><dl className="metricRecord"><dt>Screening priority</dt><dd>{a.objective_policy.priority.replaceAll("_", " ")}</dd><dt>Interpretation source</dt><dd>{a.objective_policy.basis.replaceAll("_", " ")}</dd></dl><p>{a.objective_policy.selection_instruction}</p><p>{a.chief?.objective_alignment}</p>
      <details className="auditSection"><summary>Why alternatives were not selected</summary><div className="auditBody">{(a.chief?.alternatives || []).map((v: Json) => <p key={v.candidate_id}><b>Candidate {v.candidate_id}:</b> {v.reason_not_selected}</p>)}</div></details>
      <p className="reportNote">{a.objective_policy.interpretation_note}</p></div>}
    {(selected?.temperature_deviations || []).map((d: Json) => <div className="warningRow temperatureDeviation" key={d.stage_number}><AlertTriangle size={18}/><span>Stage {d.stage_number}: batch {num(d.batch_temperature_C)} °C; screen {num(d.screen_temperature_C)} °C. {d.consequence}</span></div>)}
    <details className="auditSection"><summary>12-candidate screen and reviewer coverage</summary><div className="auditBody">
      <div className="tableScroll"><table className="dataTable"><thead><tr><th>Candidate</th><th>Stage times (min)</th><th>Stage volumes (mL)</th><th>Domain reviews</th></tr></thead><tbody>{(a.candidates || []).map((c: Json) => <tr key={c.candidate_id}><td>{c.candidate_id}{c.candidate_id === a.selected_candidate_id ? " · selected" : ""}</td><td>{c.proposal.stage_parameters.map((s: Json) => num(s.residence_time_min)).join(" + ")}</td><td>{c.proposal.stage_parameters.map((s: Json) => num(s.reactor_volume_mL)).join(" + ")}</td><td>{Object.values(a.reviews || {}).filter((rows: any) => rows.some((r: Json) => r.candidate_id === c.candidate_id)).length} / 4</td></tr>)}</tbody></table></div>
    </div></details>
    <details className="auditSection"><summary>Source evidence and analogy exclusions</summary><RecordText value={{source: a.source_context, analogies: a.analogy_audit}}/></details>
    <details className="auditSection"><summary>Limitations and next measurements</summary><RecordText value={{limitations: a.chief?.limitations, measurements: a.chief?.next_measurements}}/></details>
  </section>;
}

export function Responses({result}: {result: Json}) {
  const rows = result.result_report?.responses || [];
  const effects = result.scientific_assessment?.answer_effects;
  return <div className="reportStack"><div className="sectionTitle"><h2>Intake responses and design evidence</h2><span>{rows.length} questions</span></div>
    {effects && <p className="reportNote">Policy bindings record how inputs were routed. Council assessments are unvalidated model statements; selecting a design cannot confirm a chemical hypothesis.</p>}
    {!rows.length && <p>No standardized intake question log is stored in this run.</p>}
    {rows.map((row: Json) => <details className="auditSection responseRecord" key={row.question_id}>
      <summary><code>{row.question_id}</code><span>{row.question}</span><small>{row.assessment}</small></summary>
      <div className="auditBody"><h3>Answer · {row.status}</h3><RecordText value={row.answer ?? (row.status === "unavailable" ? "Explicitly unavailable" : null)}/>
        <h3>How it is addressed</h3><ul>{row.evidence.map((e: string, i: number) => <li key={i}>{e}</li>)}</ul>
        {effects?.questions?.filter((q: Json) => q.question_id === row.question_id).slice(-1).map((q: Json) => <p key={q.question_id}><b>Policy binding:</b> {q.effect}</p>)}
        {row.question_id === "Q-OBJ-001" && effects?.selection?.objective_alignment && <p><b>Council objective assessment, unvalidated:</b> {effects.selection.objective_alignment}</p>}
        {effects?.selection?.council_answer_assessments?.filter((q: Json) => q.question_id === row.question_id).map((q: Json, i: number) => <p key={i}><b>Council assessment, unvalidated:</b> {q.effect}</p>)}
        {row.decisions?.length > 0 && <details><summary>Related realization decisions</summary><RecordText value={row.decisions}/></details>}
        <details><summary>Normalized binding: {row.target_path}</summary><RecordText value={row.bound_value}/></details>
        <details><summary>Answer history ({row.answer_history?.length || 0} saved entries)</summary><RecordText value={row.answer_history}/></details>
      </div></details>)}
  </div>;
}

export function CouncilTranscript({result}: {result: Json}) {
  const log = result.deliberation_log || {};
  const rounds: Json[][] = (log.rounds || []).map((r: any) => Array.isArray(r) ? r : r.messages || r.entries || [r]);
  const messages = result.council_messages || [];
  const flow = result.final_design?.flow_operability;
  return <div className="reportStack"><div className="sectionTitle"><h2>Council deliberation</h2><span>{rounds.length || result.council_rounds || 0} rounds · {rounds.flat().length || messages.length} records</span></div>
    {flow?.applicable && <details className="auditSection" open><summary>Flow operability · laboratory review required</summary><div className="auditBody">
      <p>Lower gas flow does not establish protection against backflow. Numerical closure is not laboratory approval.</p>
      {flow.council_decision && <><h3>Council proposal</h3><RecordText value={flow.council_decision}/><p>Revision status: {flow.revision_status?.replaceAll("_", " ")}</p></>}
      {(flow.findings || []).map((finding: Json) => <section key={finding.finding_id}><h3>{finding.kind.replaceAll("_", " ")}</h3>
        <p>{finding.message}</p><p><b>Reverse path:</b> {finding.reverse_path?.join(" → ")}</p><p>{finding.required_action}</p>
        <p><b>Scenarios:</b> {(finding.scenarios || []).join("; ")}</p></section>)}
      <details><summary>Assessment and assumptions</summary><RecordText value={flow}/></details>
    </div></details>}
    <p className="reportNote">Recorded candidate discussion, before final inventory realization. These statements are model assessments, not final run conditions.</p>
    {!rounds.length && !messages.length && <p>No council transcript was stored in this run.</p>}
    {(rounds.length ? rounds : messages.length ? [messages] : []).map((round, i) => <details className="auditSection" key={i} open>
      <summary>Round {i + 1}<small>{round.length} recorded contributions</small></summary><div className="auditBody">
        {round.map((m: Json, j: number) => <details className="agentRecord" key={j}><summary><b>{m.agent_display_name || m.agent || m.role || `Agent ${j + 1}`}</b><span>{m.status || "recorded"}</span></summary>
          <div className="auditBody"><RecordText value={m.chain_of_thought || m.content || m.message || m.reasoning || m.value || "No narrative stored"}/>
            {Object.entries(m).filter(([k, v]) => !["agent", "agent_display_name", "status", "round"].includes(k) && v != null && v !== "" && (!Array.isArray(v) || v.length > 0)
              && !( ["chain_of_thought", "content", "message", "reasoning", "value"].includes(k) && v === (m.chain_of_thought || m.content || m.message || m.reasoning || m.value) )
            ).map(([key, value]) => <details key={key}><summary>{key.replaceAll("_", " ")}</summary><RecordText value={value}/></details>)}
          </div></details>)}
      </div></details>)}
    {Object.entries(log).filter(([k, v]) => k !== "rounds" && v != null && v !== "").map(([key, value]) => <details className="auditSection" key={key}><summary>{key.replaceAll("_", " ")}</summary><div className="auditBody"><RecordText value={value}/></div></details>)}
    {rounds.length > 0 && messages.length > 0 && <details className="auditSection"><summary>Compact council messages (original summaries)</summary><div className="auditBody"><RecordText value={messages}/></div></details>}
  </div>;
}
