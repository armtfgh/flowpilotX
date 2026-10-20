from fractions import Fraction as F
import pytest
from flora_translate.pump_settings import select_scale, setting_supported
from flora_translate.schemas import PumpSpec, FlowProposal, StreamAssignment
from flora_translate.design_realizer import _solve_liquid_stream_rates, _all_feed_devices_feasible


def pump(step=.01, minimum=.02, maximum=10):
    return PumpSpec(type='peristaltic', min_flow_rate_mL_min=minimum,
        max_flow_rate_mL_min=maximum, max_pressure_bar=10, flow_rate_increment_mL_min=step)


def test_joint_setting_preserves_stoichiometry():
    k, *_ = select_scale([F(2), F(1,2)], [pump(), pump()], .0745355)
    assert k == .08
    assert [k*2, k*.5] == [.16, .04]
    assert .04*2.1 == pytest.approx(1.05*.16*.5)


def test_different_steps_and_gas_ceiling():
    k, *_ = select_scale([F(2), F(1,2)], [pump(), pump(.02)], .0745355, maximum=.07)
    assert k == .04


def test_infeasible_setting_grid_is_not_rounded_off_ratio():
    with pytest.raises(ValueError, match='increments'):
        select_scale([F(2), F(1,2)], [pump(minimum=.05, maximum=.07), pump(minimum=.01)], .03)


def test_unknown_increment_remains_continuous():
    assert select_scale([F(1)], [pump(None)], .123456)[0] == .123456


def test_final_device_gate_rejects_off_grid():
    from flora_translate.schemas import LabInventory
    p = pump()
    proposal = FlowProposal(streams=[StreamAssignment(stream_label='A', phase='liquid',
        pump_equipment_id=p.equipment_id, flow_rate_mL_min=.149071)])
    assert not _all_feed_devices_feasible(proposal, LabInventory(pumps=[p]))
    proposal.streams[0].flow_rate_mL_min = .15
    assert _all_feed_devices_feasible(proposal, LabInventory(pumps=[p]))


def test_solver_quantizes_complete_feeds():
    p = FlowProposal(flow_rate_mL_min=.186339, streams=[
        StreamAssignment(stream_label='A', concentration_M=.5, molar_equiv=1),
        StreamAssignment(stream_label='B', concentration_M=2.1, molar_equiv=1.05)])
    issues=[]
    _solve_liquid_stream_rates(p, {'A':pump(), 'B':pump()}, [], issues)
    assert not issues
    assert [s.flow_rate_mL_min for s in p.streams] == [.16,.04]


def test_revised_workbook_modules_and_settings():
    from scripts.build_khu_inventory_20260915 import build
    from flora_translate.equipment_resources import light_fits_stage
    profile = build(write=False)
    inv=profile.lab_inventory
    assert all(r.photoreactor_module_ids == ['uv150','manual1','manual2'] for r in inv.reactors)
    assert all(p.flow_rate_increment_mL_min > 0 for p in inv.pumps)
    uv=next(l for l in inv.light_sources if l.module_id=='uv150')
    p=FlowProposal(streams=[StreamAssignment(stream_label='A',pump_equipment_id='pump_e_series_blue')])
    assert light_fits_stage(uv,next(r for r in inv.reactors if r.volume_mL==2),p,inv)
    assert not light_fits_stage(uv,next(r for r in inv.reactors if r.volume_mL==20),p,inv)
    assert not light_fits_stage(uv,inv.reactors[0],FlowProposal(),inv)


def test_gas_limit_preserves_liquid_setting_grid():
    from flora_translate.schemas import GasHardwareSpec
    from flora_translate.design_realizer import _solve_gas_stream_rates
    p=FlowProposal(flow_rate_mL_min=.2, concentration_M=.1, BPR_bar=3, streams=[
        StreamAssignment(stream_label='A',contents=['substrate'],flow_rate_mL_min=.2,concentration_M=.1,molar_equiv=1),
        StreamAssignment(stream_label='G',contents=['O2'],phase='gas',molar_equiv=2)])
    mfc=GasHardwareSpec(type='MFC',gas='O2',max_flow_sccm=.7,min_flow_sccm=.01)
    issues=[]
    _solve_gas_stream_rates(p,{'A':pump(), 'G':mfc},None,[],issues)
    assert not issues
    assert p.streams[0].flow_rate_mL_min == .15
    assert setting_supported(pump(),p.streams[0].flow_rate_mL_min)
    assert p.streams[1].gas_flow_sccm < .7
    assert p.streams[1].molar_equiv == pytest.approx(2,abs=1e-4)
