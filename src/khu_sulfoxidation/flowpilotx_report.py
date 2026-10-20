"""Review artifacts for a frozen KHU FlowPilotX profile; no laboratory release."""
from __future__ import annotations

import csv
from copy import deepcopy
from html import escape
import json
from pathlib import Path

from .provenance import digest, file_hash, write_json
from .history import _load_private_context


def condition_rows(profile, selected_ids=None):
    ids = selected_ids if selected_ids is not None else profile['default_selected_candidate_ids']
    by_id = {c['id']:c for c in profile['candidates']}
    if not ids or len(ids)!=len(set(ids)) or any(i not in by_id for i in ids):
        raise ValueError('Nonempty unique candidate selection required')
    rows=[]
    for ident in ids:
        c=by_id[ident]
        p,calc=c['parameters'],c['calculated']
        rows.append({'id':ident,'target_nominal_transit_min':p['target_time_min'],
            'calculated_nominal_transit_min':p['transit_min'],'target_inlet_O2_eq':p['target_oxygen_eq'],
            'calculated_inlet_O2_eq':p['oxygen_eq'],'proposed_liquid_mL_min':p['liquid_mL_min'],
            'proposed_pure_O2_STP_mL_min':p['oxygen_STP_mL_min'],
            'calculated_O2_mmol_min':calc['oxygen_mmol_min_supplied'],
            'central_plug_substrate_mmol_min':calc['substrate_mmol_min'],
            'central_plug_catalyst_mmol_min':calc['catalyst_mmol_min'],
            'calculated_gas_reactor_mL_min':calc['gas_reactor_flow_mL_min_estimate'],
            'liquid_only_space_time_min':calc['liquid_only_space_time_min'],
            'no_slip_liquid_holdup_mL_estimate':p['coil_mL']*p['liquid_mL_min']/(p['liquid_mL_min']+calc['gas_reactor_flow_mL_min_estimate']),
            'nominal_coil_mL':p['coil_mL'],'proposed_temperature_C':p['temperature_C_proposed'],
            'BPR_inventory_id':c['proposed_setpoints']['bpr_equipment_id'],
            'BPR_label_bar':c['proposed_setpoints']['bpr_label_value_bar'],
            'BPR_basis_assumption':'gauge','assumed_reactor_pressure_bar_abs':p['pressure_bar_abs_assumed'],
            'assumed_STP_temperature_K':calc['gas_reference_temperature_K_assumed'],
            'assumed_STP_pressure_bar_abs':calc['gas_reference_pressure_bar_abs_assumed'],
            'assumed_O2_mole_fraction':calc['gas_oxygen_mole_fraction_assumed'],
            'feed_mode':'sample_loop','loop_volume_mL':None,'collection_window_min':None,
            'status':'conditional_not_laboratory_released','question':c['question']})
    return rows


def _csv(path, rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def results_template(profile, selected_ids):
    by_id={c['id']:c for c in profile['candidates']}
    rows=[]
    for ident in selected_ids:
        rows.append({'experiment_id':ident,'proposal_id':ident,'repeat_of':None,
            'proposed':deepcopy(by_id[ident]['parameters']),
            'actual_conditions':{k:None for k in ['date','operator','substrate_identity','solvent','concentration_M','catalyst_loading_mol_percent',
                'loop_inventory_id','loop_volume_mL','injected_sample_volume_mL','valve_identity_port_map','carrier_flow_mL_min','oxygen_flow_reference_mL_min',
                'oxygen_purity','MFC_calibration','gas_reference_temperature_K','gas_reference_pressure_bar_abs','reactor_pressure_bar_abs',
                'BPR_basis','temperature_C','light_setting','injection_timestamp','fraction_start_min','fraction_end_min','fraction_volume_mL','flush_volume_mL','sample_handling']},
            'analytical_results':{k:None for k in ['residual_sulfide_pct','sulfoxide_pct','sulfone_pct','other_products_pct','unknown_components_pct',
                'analytical_method','internal_standard','calibration','analytical_uncertainty','reported_component_sum_pct','raw_spectrum_id']},
            'observations':{k:None for k in ['bubbles_flow_pattern','pressure_instability','precipitation','discoloration','sample_dilution','comments']}})
        rows[-1]['actual_conditions'].update(deepcopy(profile.get('observation_contract',{}).get('actual_conditions_template',{})))
    repeat=profile['optional_repeat']
    if repeat['repeat_of'] in by_id:
        original=next((r for r in rows if r['proposal_id']==repeat['repeat_of']),None)
        if original is not None:
            row=deepcopy(original)
            row.update(experiment_id=repeat['id'],repeat_of=repeat['repeat_of'])
            row['optional']=True
            rows.append(row)
    return {'schema_version':'khu-flowpilotx-results-template-v1','status':'blank_unmeasured',
        'observation_contract':deepcopy(profile.get('observation_contract')),
        'rules':['Keep reported percentages including values above100; no clipping or mass-closure normalization.',
            'Keep batch and flow modes separate. Use separate fraction rows when applicable.',
            'Do not overwrite proposals with actual settings. A blank observation is unknown, not false.'], 'experiments':rows}


def export_review(profile, output_dir, selected_ids=None):
    """Write immutable JSON/CSV/HTML around a frozen selection; diagrams separate."""
    private_display = _load_private_context()["review_display"]
    out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    rows=condition_rows(profile,selected_ids)
    ids=[r['id'] for r in rows]
    for name in ['profile.json','conditions.csv','review.html','results_template.json','feed_preparation.json','calculation_basis.json','selection.json','package_manifest.json']:
        if (out/name).exists(): raise FileExistsError('Use a new package version; existing artifacts preserved')
    write_json(out/'profile.json',profile)
    write_json(out/'selection.json',{'selected_candidate_ids':ids,'profile_sha256':digest(profile),
        'status':'offline_preferred_grid' if selected_ids is None else 'externally_frozen_selection',
        'optional_repeat':profile['optional_repeat'],'laboratory_released':False})
    _csv(out/'conditions.csv',rows)
    write_json(out/'feed_preparation.json',profile['materials'])
    write_json(out/'calculation_basis.json',profile['calculation_basis'])
    write_json(out/'results_template.json',results_template(profile,ids))
    all_attempts=[*profile['candidate_attempts'],*[r for a in profile['engineering_alternatives'] for r in a['attempts']]]
    write_json(out/'candidate_attempts.json',all_attempts)
    write_json(out/'history_omitted_request.json',profile['history_omitted_request_fixture'])
    by_id={c['id']:c for c in profile['candidates']}
    questions={q['id']:q for q in profile['questions']}
    esc=lambda x:escape(str(x))
    tr=''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [r['id'],f"{r['target_nominal_transit_min']:g}",f"{r['calculated_nominal_transit_min']:.2f}",
        f"{r['target_inlet_O2_eq']:g}",f"{r['calculated_inlet_O2_eq']:.3f}",f"{r['proposed_liquid_mL_min']:.2f}",f"{r['proposed_pure_O2_STP_mL_min']:.2f}"] )+'</tr>' for r in rows)
    list_html=lambda items:'<ul>'+''.join('<li>'+esc(x)+'</li>' for x in items)+'</ul>'
    cards=[]
    for r in rows:
        c=by_id[r['id']];q=questions[c['question_ids'][0]]
        cards.append(f"<section><h2>{esc(r['id'])}: {esc(c['design_role'])}</h2><p>{esc(q['question'])}</p>"
            f"<img src='diagrams/{esc(r['id'])}.svg' alt='Conditional sample-loop process schematic'>"
            f"<p>{esc(c['rationale'])}</p><h3>Possible observations</h3>{list_html(q['expected_observations'])}"
            f"<h3>Next decision</h3>{list_html(q['decision_rules'])}<p class='small'>Evidence IDs: {esc(', '.join(c['evidence_ids']))}</p></section>")
    material=profile['materials']['sample_stock']
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>KHU FlowPilotX conditional proposals</title>
<style>body{{font:16px/1.5 Arial,sans-serif;color:#17263b;max-width:1180px;margin:45px auto;padding:0 24px}}h1,h2{{color:#143e59}}section{{border-top:2px solid #dce8ed;margin-top:34px;padding-top:20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d8e2e9;text-align:left}}th{{background:#eaf3f7}}img{{max-width:100%;height:auto}}.notice{{padding:18px;background:#fff5d8;border-left:5px solid #bd831c}}.small{{font-size:12px;overflow-wrap:anywhere}}code{{font-size:14px}}@media print{{section{{break-inside:avoid}}}}</style>
<h1>KHU FlowPilotX: pure oxygen + sample-loop screen</h1>
<p class="notice"><b>Conditional review package; not laboratory released.</b> One proposed assembly, {len(rows)} unique sequential conditions. No flow outcomes, predicted yields or actual residence measurements are claimed. Installed loop size, valve routing, calibration, pressure basis and collection remain unresolved.</p>
{private_display["apparatus_html"]}
<table><tr><th>ID</th><th>Target nominal min</th><th>Calculated nominal min</th><th>Target O2 eq</th><th>Calculated O2 eq</th><th>Liquid mL/min</th><th>PureO2 STP mL/min</th></tr>{tr}</table>
<p>Optional independent repeat: KX-R01 of KX04, additional to the unique count; fresh preparation/injection and operation, no runtime or execution order assigned. If KX04 is absent from a final selected set, the repeat requires an explicit selection review.</p>
<h2>Why this window and coil?</h2>{list_html(profile['configuration']['rationale'].values())}
<p>The primary-literature review includes an estimated21min thioanisole/PDI-bead result at0.2M in oxygen-enriched EtOH (<a href="https://pubs.acs.org/doi/10.1021/acs.oprd.4c00256">Ali et al.,2024</a>). Different catalyst, bed geometry and time basis make this contextual support for exploration, not acridine flow kinetics. No matching neutral-acridine flow validation is claimed.</p>
<h2>Calculation basis</h2><p>R={profile['calculation_basis']['R']:.14f} mL bar/(mmol K). M and mL/min give mmol/min.</p>
<p><code>n_substrate=C*Q_L; n_O2=xO2*P_ref*Q_ref/(R*T_ref); inletO2eq=n_O2/n_substrate.<br>Q_g,reactor=Q_ref*(P_ref/P_abs)*(T_reactor/T_ref); nominal transit=V_coil/(Q_L+Q_g,reactor).</code></p>
<p>Gas compression changes the transit estimate, not inlet molar equivalents. Controller settings are on the reported0.01mL/min/STP-mL/min grids, with both quantities recalculated and within3% of targets. V/Q_L is a liquid-only bookkeeping index, not residence time. Ideal gas, uniform pressure, no slip and constant inlet gas are assumptions; uptake, solvent vapor, dissolution and RTD are unmeasured.</p>
<h2>Finite-loop and material basis</h2><p>At the undiluted plug center the table gives nominal substrate/O2 flow ratios. Front/tail dilution changes local substrate concentration and effective O2 excess. The loop volume, nominal injection duration V_loop/Q_L, useful plateau, arrival and collection window cannot be assigned yet. Modeled no-slip liquid holdup is not a measurement or a minimum proven loop volume. Gas-on time, carrier/flush consumption and total campaign material demand remain unknown.</p>
<p>Illustrative10mL final stock: {material['illustrative_substrate_mmol']:g}mmol thioanisole ({material['illustrative_substrate_pure_equivalent_mg']:.2f}mg pure-equivalent) and {material['illustrative_catalyst_mmol']:g}mmol acridine ({material['illustrative_catalyst_pure_equivalent_mg']:.4f}mg pure-equivalent). Actual weighed mass=pure-equivalent mass/actual purity fraction. Confirm identity, purity and solubility, then make to final volume with EtOH. This is a scalable stock example, not installed loop volume or total supply. MW124.20/179.22g/mol: sources and formulas are in feed_preparation.json. {private_display["available_material_text"]}</p>
<h2>Release information still needed</h2>{list_html(r['description'] for r in profile['requirements'] if r['scope']=='release')}
{private_display["evidence_html"]}
<p>All12 frozen candidate proposals, {len(all_attempts)} solver attempts across selected/alternative coils, rejected grid intervals, configurations and source hashes remain in profile.json/candidate_attempts.json. Feasible unselected candidates are not chemical rejections. No new history-omitted nominations have been fabricated; the optional matched request fixture specifies a future information boundary.</p>
{''.join(cards)}<h2>Limits</h2>{list_html(profile['limitations'])}<p class="small">Authoritative profile SHA256:{digest(profile)}</p></html>'''
    (out/'review.html').write_text(html,encoding='utf-8')
    manifest={'schema_version':'khu-flowpilotx-review-manifest-v1','profile_sha256':digest(profile),'selected_candidate_ids':ids,
        'source_hashes':profile['source_hashes'],'artifacts':{p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()},
        'laboratory_released':False,'all_prior_outputs_preserved':True}
    write_json(out/'package_manifest.json',manifest)
    return manifest
