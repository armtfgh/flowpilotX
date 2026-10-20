import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest

from flora_design.visualizer import flowsheet_builder as renderer
from flora_translate.schemas import ProcessTopology, UnitOperation, StreamConnection

NS = {'s':'http://www.w3.org/2000/svg'}
FIXTURES = json.loads((Path(__file__).parent/'fixtures/diagram_stage0_topologies.json').read_text())


def render_check(topology, tmp_path):
    original = topology.model_dump_json()
    builder = renderer.FlowsheetBuilder()
    svg, png = builder.build(topology, 'Layout regression', str(tmp_path/'process.svg'), str(tmp_path/'process.png'))
    assert builder.last_render_info['renderer'] == 'graphviz'
    assert topology.model_dump_json() == original
    if not topology.unit_operations:
        assert (svg,png)==('','')
        return None,builder
    assert Path(png).stat().st_size > 1000
    root = ET.parse(svg)
    positions = {}
    for node in root.findall('.//s:g[@class="node"]', NS):
        img = node.find('s:image', NS)
        if img is not None:
            positions[node.find('s:title', NS).text] = (
                float(img.get('x')), float(img.get('y')) + float(img.get('height').removesuffix('px'))/2)
    main = [positions[key.replace('-','_').replace(' ','_')] for key in builder.last_render_info['main_process_path']
            if key.replace('-','_').replace(' ','_') in positions]
    assert len(main) >= 2
    assert max(y for x,y in main) - min(y for x,y in main) <= 2, main
    assert all(a[0] < b[0] for a,b in zip(main, main[1:]))
    return root, builder


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda f:f['name'])
def test_existing_topologies_have_horizontal_main_icons(fixture, tmp_path):
    render_check(ProcessTopology.model_validate(fixture['topology']), tmp_path)


def branching_topology():
    operations = [UnitOperation(op_id='feed_a',op_type='pump',parameters={
        'contents':['A long substrate name and concentration that wraps over many caption lines (0.1 M)'],
        'flow_rate_mL_min':.02}), UnitOperation(op_id='reactor_1',op_type='photoreactor'),
        UnitOperation(op_id='gas',op_type='mfc',parameters={'contents':['O2']}),
        UnitOperation(op_id='cv',op_type='check_valve',label='CV-3301'),
        UnitOperation(op_id='mix',op_type='mixer'),UnitOperation(op_id='reactor_2',op_type='photoreactor'),
        UnitOperation(op_id='product',op_type='collector')]
    pairs=[('feed_a','reactor_1','liquid'),('reactor_1','mix','liquid'),('gas','cv','gas'),
           ('cv','mix','gas'),('mix','reactor_2','gas-liquid'),('reactor_2','product','liquid')]
    return ProcessTopology(unit_operations=operations,streams=[StreamConnection(from_op=a,to_op=b,stream_type=c) for a,b,c in pairs])


def test_gas_check_valve_remains_a_branch(tmp_path):
    topology = branching_topology()
    root, builder = render_check(topology, tmp_path)
    assert builder.last_render_info['main_process_path'] == ['feed_a','reactor_1','mix','reactor_2','product']
    assert len(root.findall('.//s:g[@class="edge"]', NS)) == len(topology.streams)


def test_long_title_is_wrapped_without_truncation(tmp_path):
    title='A detailed connected process title '*5+'FINAL QUALIFIER'
    svg,_=renderer.FlowsheetBuilder().build(branching_topology(),title,str(tmp_path/'p.svg'),str(tmp_path/'p.png'))
    root=ET.parse(svg)
    text=' '.join(n.text or '' for n in root.findall('.//s:text',NS))
    assert title.strip() in text


def test_uploaded_check_valve_icon_is_loaded_without_code_changes(tmp_path, monkeypatch):
    icons=tmp_path/'icons'
    shutil.copytree(renderer.ICONS_DIR,icons)
    # Stand-in file only inside this test's temporary directory.
    shutil.copy2(icons/'bpr.png',icons/'check_valve.png')
    monkeypatch.setattr(renderer,'ICONS_DIR',icons)
    root, _ = render_check(branching_topology(),tmp_path)
    cv=next(n for n in root.findall('.//s:g[@class="node"]',NS) if n.find('s:title',NS).text=='cv')
    assert cv.find('s:image',NS).get('{http://www.w3.org/1999/xlink}href')=='check_valve.png'


@pytest.mark.parametrize('feed_count',[3,4,5])
def test_multiple_feeds_preserve_all_edges_without_invented_mixers(tmp_path, feed_count):
    ops=[UnitOperation(op_id=f'feed{i}',op_type='pump',parameters={'contents':[f'Reagent {i}']}) for i in range(feed_count)]
    ops += [UnitOperation(op_id='mix',op_type='mixer'),UnitOperation(op_id='reactor',op_type='coil_reactor'),UnitOperation(op_id='product',op_type='collector')]
    streams=[StreamConnection(from_op=f'feed{i}',to_op='mix',stream_type='liquid') for i in range(feed_count)]
    streams += [StreamConnection(from_op='mix',to_op='reactor',stream_type='liquid'),StreamConnection(from_op='reactor',to_op='product',stream_type='liquid')]
    root,_=render_check(ProcessTopology(unit_operations=ops,streams=streams),tmp_path)
    assert len(root.findall('.//s:g[@class="edge"]',NS))==len(streams)
    assert len(root.findall('.//s:g[@class="node"]',NS))==len(ops)
