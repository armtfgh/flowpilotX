"""Render the new pure-O2/sample-loop adapter with preserved FlowPilot assets."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import types
import xml.etree.ElementTree as ET

from .topology_figures import BUILDER, ARTIFACTS, ICONS, _load_file
from .flowpilotx_profile import EDGES
from .provenance import digest, file_hash, write_json


def render_figures(project_root, profile, output_dir, selected_ids=None):
    root, out = Path(project_root).resolve(), Path(output_dir).resolve()
    for name in ["diagrams", "render_runs", "assets", "renderer_source", "render_index.json"]:
        if (out/name).exists():
            raise FileExistsError("Use a new package version; existing render artifacts are immutable")
    before = {str(p.relative_to(root)): file_hash(p) for p in [root/BUILDER,root/ARTIFACTS,*sorted((root/ICONS).glob('*'))] if p.is_file()}
    out.mkdir(parents=True,exist_ok=True)
    shutil.copytree(root/ICONS,out/'assets/icons')
    (out/'renderer_source').mkdir()
    for source in [BUILDER,ARTIFACTS]:
        shutil.copy2(root/source,out/'renderer_source'/Path(source).name)
    renderer = _load_file(root/BUILDER,'_khu_flowpilotx_original_renderer')
    artifacts = _load_file(root/ARTIFACTS,'_khu_flowpilotx_original_artifacts')
    renderer.ICONS_DIR = out/'assets/icons'
    renderer.ASSETS = {k:renderer.ICONS_DIR/v.name for k,v in renderer.ASSETS.items()}
    base_reactor, base_box = renderer._reactor_label, renderer._add_textbox
    def reactor(op):
        p=op.parameters
        return base_reactor(op)+f"\nNominal gas-liquid transit\n{p['estimated_mixture_transit_min']:.2f} min (calculated)\nTarget: {p['target_mixture_transit_min']:g} min"
    def mfc(op):
        p=op.parameters
        return '<BR/>'.join([renderer._font_lines('Pure O2 / FF-C00',bold=True),
            renderer._font_lines(f"O2: {p['gas_flow_sccm']:.2f} STP mL/min",10,'#B91C1C'),
            renderer._font_lines(f"Inlet O2: {p['inlet_oxygen_equivalents']:.3f} eq",10),
            renderer._font_lines(f"Assumed STP: {p['reference_temperature_K']:.2f} K\n{p['reference_pressure_bar_abs']:.5f} bar abs",8),
            renderer._font_lines('O2 calibration / supply pending',8)])
    def bpr(op):
        p=op.parameters
        return f"Fixed BPR: {p['pressure_bar']:g} bar\nGauge basis assumed\nModel: {p['modeled_reactor_pressure_bar_abs']:.5f} bar abs"
    def pump(op):
        return '<BR/>'.join([renderer._font_lines('EtOH carrier / R-series',bold=True),
            renderer._font_lines(f"{op.parameters['flow_rate_mL_min']:.2f} mL/min",10,'#1D4ED8'),
            renderer._font_lines('Drives finite sample injection',8)])
    def box(dot,node_id,op):
        if op.op_type in {'injection_valve','sample_loop'}:
            # Graphviz plain nodes ignore node-level dash styles; the HTML
            # table itself must carry STYLE=DASHED. Keep original named ports.
            lines=renderer._wrap(renderer._textbox_label(op)).split('\n')
            rows=''.join(f'<TR><TD ALIGN="CENTER" WIDTH="145"><FONT POINT-SIZE="9" COLOR="#1E293B">{renderer._esc(line)}</FONT></TD></TR>' for line in lines)
            label=('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0"><TR>'
                '<TD PORT="inp" WIDTH="1" HEIGHT="1"></TD><TD>'
                '<TABLE BORDER="1" STYLE="DASHED" COLOR="#B45309" CELLBORDER="0" CELLSPACING="0" CELLPADDING="8" BGCOLOR="#FFFAED">'
                +rows+'</TABLE></TD><TD PORT="out" WIDTH="1" HEIGHT="1"></TD></TR></TABLE>>')
            dot.node(node_id,label=label,shape='plain')
        else:
            base_box(dot,node_id,op)
    renderer._reactor_label,renderer._mfc_label,renderer._bpr_label,renderer._pump_label = reactor,mfc,bpr,pump
    renderer._add_textbox = box
    old_cache=os.environ.get('XDG_CACHE_HOME')
    os.environ['XDG_CACHE_HOME']=str(out/'render_cache')
    cases=[]
    try:
        if not renderer._ensure_graphviz_on_path():
            raise RuntimeError('Graphviz required for original FlowPilot figures')
        class Builder(renderer.FlowsheetBuilder):
            def build(self,data,**kwargs):
                obj=types.SimpleNamespace(unit_operations=[types.SimpleNamespace(**deepcopy(o)) for o in data['unit_operations']],
                    streams=[types.SimpleNamespace(**deepcopy(s)) for s in data['streams']])
                return super().build(obj,**kwargs)
            def _legacy_fallback(self,*args,**kwargs):
                raise RuntimeError('Diagnostic fallback is not a deliverable')
        (out/'diagrams').mkdir()
        selected_ids = selected_ids or profile['default_selected_candidate_ids']
        by_id = {c['id']:c for c in profile['candidates']}
        if len(selected_ids)!=len(set(selected_ids)) or any(x not in by_id for x in selected_ids):
            raise ValueError('Unique known candidate IDs required')
        for candidate in [by_id[x] for x in selected_ids]:
            topology=deepcopy(candidate['diagram'])
            if topology['source_parameters_sha256'] != digest(candidate['parameters']):
                raise ValueError('Diagram differs from authoritative condition')
            p=candidate['parameters']
            title=f"{candidate['id']} - {p['target_time_min']:g} min target / {p['target_oxygen_eq']:g} inlet O2 eq"
            builder=Builder()
            result=artifacts.render_topology_artifacts(topology,title=title,base_dir=out/'render_runs'/candidate['id'],builder=builder)
            if result['renderer']!='graphviz' or result['render_status']!='complete' or result['warnings']:
                raise ValueError(f"Original renderer did not complete: {result}")
            svg=ET.parse(result['svg_path']).getroot()
            images=list(svg.iter('{http://www.w3.org/2000/svg}image'))
            if len(images)!=7 or any(not im.get('{http://www.w3.org/1999/xlink}href',im.get('href','')).startswith('data:image/') for im in images):
                raise ValueError('Expected seven embedded original equipment icons')
            edge_titles=[g.find('{http://www.w3.org/2000/svg}title').text for g in svg.iter('{http://www.w3.org/2000/svg}g') if g.get('class')=='edge']
            actual={'->'.join(part.split(':',1)[0] for part in e.split('->')) for e in edge_titles}
            if actual!={a+'->'+b for a,b,_ in EDGES}:
                raise ValueError('Rendered process connectivity differs from finite-loop schematic')
            files={}
            for key,ext,name in [('svg_path','svg','svg'),('png_path','png','png'),('topology_path','json','topology_json')]:
                dest=out/'diagrams'/f"{candidate['id']}.{ext}"
                shutil.copyfile(result[key],dest)
                files[name]=str(dest.relative_to(out))
            cases.append({'condition_id':candidate['id'],**files,'renderer':result['renderer'],
                'source_parameters_sha256':digest(candidate['parameters']),
                'main_process_path':builder.last_render_info['main_process_path'],
                'render_manifest':str(Path(result['render_manifest_path']).relative_to(out))})
        if any(file_hash(root/p)!=h for p,h in before.items()):
            raise ValueError('Baseline renderer/icons changed')
        index={'schema_version':'khu-flowpilotx-figures-v1','cases':cases,'profile_sha256':digest(profile),
            'original_renderer_and_icon_hashes':before,'baseline_modified':False,
            'overrides':['Pure-O2/STP MFC caption','Carrier rather than substrate pump caption','Explicit nominal mixture time',
                'Assumed BPR pressure basis','Unresolved injection/loop textbox labels with dashed HTML borders'],
            'scope':'INJECT-state conceptual routing only; unresolved hardware is not a confirmed P&ID or control sequence'}
        write_json(out/'render_index.json',index)
        return index
    finally:
        if old_cache is None: os.environ.pop('XDG_CACHE_HOME',None)
        else: os.environ['XDG_CACHE_HOME']=old_cache
