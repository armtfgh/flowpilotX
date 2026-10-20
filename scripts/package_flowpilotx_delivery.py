#!/usr/bin/env python3
"""Create a local review bundle; never uploads or changes the source packages."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/flowpilotx_delivery_2026-10-15_v1'
OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
folders={
 'KHU_review':ROOT/'outputs/flowpilotx_khu_2026-10-15_v1',
 'Presentation':ROOT/'outputs/flowpilotx_presentation_2026-10-15_v1',
 'GUI_screenshots':ROOT/'outputs/flowpilotx_gui_2026-10-15_v1_highres'}
extras={
 'Evaluation/requirements_status.json':ROOT/'outputs/flowpilotx_platform_2026-10-15_v1/requirements_status.json',
 'Evaluation/requirements_review.html':ROOT/'outputs/flowpilotx_platform_2026-10-15_v1/requirements_review.html',
 'Evaluation/benchmark_report.json':ROOT/'outputs/flowpilotx_platform_2026-10-15_v1/benchmark_report.json',
 'Evaluation/scientific_audit.json':ROOT/'review/flowpilotx_qwen_scientific_audit_2026-10-15_v2.json',
 'Evaluation/independent_package_audit.json':ROOT/'review/flowpilotx_final_package_audit_2026-10-15_v1.json',
 'Evaluation/requirements_audit.json':ROOT/'review/flowpilotx_requirements_audit_2026-10-15_v2.json',
 'Evaluation/software_tests.txt':ROOT/'tmp/flowpilotx_final_tests_v2.txt',
 'Evaluation/handoff_integrity.json':ROOT/'tmp/flowpilotx_final_handoff.json',
 'Evaluation/browser_workflows.json':ROOT/'tmp/flowpilotx_gui_qa/browser_checks.json',
 'Evaluation/browser_proposal_contracts.json':ROOT/'tmp/flowpilotx_gui_qa/proposal_contract_checks.json',
 'FLOWPILOTX_PLATFORM.md':ROOT/'docs/FLOWPILOTX_PLATFORM.md'}
files={**{f'{prefix}/{p.relative_to(folder)}':p for prefix,folder in folders.items() for p in folder.rglob('*') if p.is_file()},**extras}
quality={'schema_version':'flowpilotx-delivery-quality-v1','status':'software_verified_conditional_chemistry',
 'tests':{'run':215,'passed':214,'skipped':1,'focused_flowpilotx':45},
 'preservation':{'handoff_files_unchanged':820,'old_ppt_sha256':sha(ROOT/'outputs/khu_presentation_2026-10-14_v8/KHU_homogeneous_flow_screen.pptx'),
 'old_ppt_matches_recorded_hash':sha(ROOT/'outputs/khu_presentation_2026-10-14_v8/KHU_homogeneous_flow_screen.pptx')=='3e2bd57eadc147aba040e2b79c38345bfce25c881506670db7856d6070332748'},
 'independent_package_checks':{'passed':47,'total':47},'presentation':{'slides':22,'all_visually_checked':True,'layout_warnings':0},
 'browser':{'isolated_workflows_and_contract_checks':17,'production_read_only_quota_and_saved_round_checks':True,'screenshot_device_scale_factor':4,'external_requests_in_capture':0},
 'qwen':{'attempts':12,'transport_json_successes':10,'generation_failures':2,'paid_api_cost_usd':0,'local_compute_cost':'not metered',
 'software_check_summary':'8/10 recorded checks; includes explicit post-hoc synthetic-fixture repair, not a statistical model accuracy measure',
 'scientific_reasoning':'failed, including assisted Chief; selection settings independently checked; unsupported model hypotheses unapproved'},
 'requirements':{'tracked':40,'tested':27,'conditional':12,'deferred':1,'scientific_questions_unresolved':12,'lab_release_conditions_unresolved':8},
 'laboratory_flow_results':0,'laboratory_released':False,
 'implementation_note':'The frozen final proposal correctly records generation-time source hashes. Subsequent evidence.py/selection.py changes tightened unknown analytical-method qualification; no numerical design, source evidence or KHU outcome changed. Production server restarted with that guard.',
 'current_source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/flowpilotx').glob('*.py')},
 'files':{name:sha(p) for name,p in sorted(files.items())}}
(OUT/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
readme='''FlowPilotX / KHU review delivery —15October2026

Open Presentation/FlowPilotX_KHU.pptx or KHU_review/review.html.
KHU_review/conditions.csv contains six settings; KX04 has one separate independent repeat.
KHU_review/results_import_template.json is the blank GUI import form; RESULTS_README.txt explains the detailed laboratory worksheet.
Evaluation/requirements_review.html tracks all40requests and12unresolved scientific questions.
GUI_screenshots contains native4x lossless PNGs; open originals for full resolution.

This is a conditional review package, not a released laboratory procedure or demonstrated high-yield result.
Qwen selected the six setting IDs, but failed scientific reasoning review. Reviewed adapter questions accompany the physically checked settings. All raw model attempts remain in the workspace under outputs/flowpilotx_platform_2026-10-15_v1/model_calls; no claim of autonomous scientific correctness.

The general GUI is running locally at http://127.0.0.1:8788. Restart in the workspace with .venv/bin/python scripts/run_flowpilotx_platform.py --port 8788.
The12-attempt Qwen benchmark allowance is exhausted; offline selection remains available explicitly. No paid fallback is enabled.
'''
(OUT/'README.txt').write_text(readme)
archive=OUT/'FlowPilotX_KHU_review_bundle.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name,p in sorted(files.items()):z.write(p,name)
 z.write(OUT/'quality.json','quality.json');z.write(OUT/'README.txt','README.txt')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
(OUT/'archive_manifest.json').write_text(json.dumps({'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'entry_count':len(files)+2},indent=2)+'\n')
print(json.dumps({'archive':str(archive),'megabytes':round(archive.stat().st_size/1024**2,2),'entries':len(files)+2},indent=2))
