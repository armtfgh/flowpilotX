from flora_design.visualizer.flowsheet_builder import (
    ASSETS,
    _separator_asset,
    _separator_kind,
)
from flora_translate.schemas import UnitOperation


def _separator(*, op_type="phase_separator", label="", parameters=None):
    return UnitOperation(
        op_id="separator_1",
        op_type=op_type,
        label=label,
        parameters=parameters or {},
    )


def test_gas_liquid_phase_metadata_selects_existing_gl_icon():
    operation = _separator(parameters={"phases": ["gas", "liquid"]})

    assert _separator_kind(operation) == "gas_liquid"
    assert _separator_asset(operation) == ASSETS["g_l_separator"]
    assert _separator_asset(operation).is_file()


def test_liquid_liquid_phase_metadata_selects_existing_ll_icon():
    operation = _separator(parameters={"phases": ["liquid", "liquid"]})

    assert _separator_kind(operation) == "liquid_liquid"
    assert _separator_asset(operation) == ASSETS["l_l_separator"]
    assert _separator_asset(operation).is_file()


def test_separator_aliases_override_ambiguous_labels():
    gas_liquid = _separator(op_type="gas_liquid_separator", label="Separator")
    liquid_liquid = _separator(op_type="liq_liq_extraction", label="Separator")

    assert _separator_asset(gas_liquid) == ASSETS["g_l_separator"]
    assert _separator_asset(liquid_liquid) == ASSETS["l_l_separator"]


def test_adjacent_gas_liquid_stream_selects_gl_icon_when_parameters_are_missing():
    operation = _separator(label="Phase separator")

    assert _separator_kind(operation, ["gas-liquid", "liquid"]) == "gas_liquid"
