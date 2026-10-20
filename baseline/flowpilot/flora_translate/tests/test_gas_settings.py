from copy import deepcopy
import pytest

from flora_translate.gas_settings import gas_setting_at_least, gas_setting_supported, maximum_gas_setting
from flora_translate.schemas import GasHardwareSpec, FlowProposal


def device(**kw):
    return GasHardwareSpec(type="MFC", gas="O2", min_flow_sccm=0.01,
        max_flow_sccm=kw.pop("maximum", 10), flow_rate_increment_sccm=kw.pop("step", .01), **kw)


def test_grid_is_distinct_from_minimum_and_never_rounds_dose_down():
    d = device()
    assert gas_setting_at_least(d, .089651) == .09
    assert gas_setting_at_least(d, .426909) == .43
    assert gas_setting_at_least(d, .09) == .09
    assert gas_setting_at_least(d, .09 + 1e-16) == .09
    assert gas_setting_at_least(d, .001) == .01
    assert gas_setting_supported(d, .09)
    assert not gas_setting_supported(d, .089651)


def test_grid_maximum_and_impossible_ranges():
    assert maximum_gas_setting(device(maximum=.095)) == .09
    with pytest.raises(ValueError, match="no feasible MFC"):
        gas_setting_at_least(device(maximum=.095), .091)
    with pytest.raises(ValueError, match="no positive"):
        maximum_gas_setting(device(maximum=.009))
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            device(step=bad)


def test_unspecified_resolution_preserves_old_behavior():
    d = device(step=None)
    assert gas_setting_at_least(d, .089651) == .089651
    assert gas_setting_supported(d, .089651)


def test_public_realization_and_candidate_reclosure_do_not_drift():
    from flora_translate.tests.test_scientific_gas import photo_case
    from flora_translate.engine.council_v4.scientific import build_screen_pool, signature
    from flora_translate.design_realizer import realize_executable_design, _all_feed_devices_feasible
    from flora_translate.residence_time_basis import gas_equiv_from_stp_flow
    p, batch, plan, inv = photo_case()
    inv.gas_hardware[0].flow_rate_increment_sccm = .01
    pool, _ = build_screen_pool(p, batch, plan, inv)
    assert len(pool) == 12
    for c in pool:
        prop = FlowProposal.model_validate(c["proposal"])
        g = next(s for s in prop.streams if s.phase == "gas")
        l = next(s for s in prop.streams if s.phase == "liquid")
        assert gas_setting_supported(inv.gas_hardware[0], g.gas_flow_sccm)
        assert g.molar_equiv == pytest.approx(gas_equiv_from_stp_flow(g.gas_flow_sccm, l.flow_rate_mL_min, l.concentration_M, g.gas_reagent_mole_fraction), abs=1e-4)
        revised, _, validation = realize_executable_design(prop, batch_record=batch,
            chemistry_plan=plan.model_copy(deep=True), inventory=inv)
        assert signature(revised) == signature(prop)
        assert all(validation["checks"].values())
        offgrid = deepcopy(prop)
        next(s for s in offgrid.streams if s.phase == "gas").gas_flow_sccm += .001
        assert not _all_feed_devices_feasible(offgrid, inv)
