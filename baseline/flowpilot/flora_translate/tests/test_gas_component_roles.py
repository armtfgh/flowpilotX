import pytest
from flora_translate.executable_artifacts import _matching_role, _normalize_name, _stream_components


@pytest.mark.parametrize("formula, name", [("H2", "Hydrogen"), ("H₂", "Molecular hydrogen"),
    ("O2", "Oxygen"), ("N2", "Nitrogen"), ("CO2", "Carbon dioxide"),
    ("CO", "Carbon monoxide"), ("D2", "Deuterium")])
def test_formula_preserves_the_declared_component_role_and_numbers(formula, name):
    result = {"chemistry_plan": {"reagents": [{"name": name, "role": "Declared reagent role"}]}}
    stream = {"stream_label": "G", "contents": [formula], "phase": "gas",
        "molar_equiv": 3.5411, "gas_active_equivalents": 3.5411, "gas_flow_sccm": 1.0,
        "gas_reagent_mole_fraction": 1.0, "gas_active_molar_flow_mmol_min": .044617516}
    component = _stream_components(result, [stream])[0]
    assert component.role == "declared reagent role"
    assert component.molar_equiv == pytest.approx(3.5411)
    assert stream["gas_flow_sccm"] == 1.0


@pytest.mark.parametrize("query, declared", [("H2", "H2O2"), ("Hydrogen", "Hydrogen peroxide"),
    ("O2", "Air"), ("CO", "CO2"), ("H2", "D2"), ("H2O2", "H2"),
    ("Hydrogen peroxide", "Hydrogen"), ("Air", "O2")])
def test_gas_alias_does_not_transfer_a_role_from_a_different_species(query, declared):
    assert _matching_role(_normalize_name(query), {_normalize_name(declared): {"role": "wrong species"}}) is None


def test_gas_alias_with_conflicting_plan_declarations_stays_unresolved():
    assert _matching_role("h2", {"hydrogen": {"role": "reductant"},
        "molecular hydrogen": {"role": "purge gas"}}) is None


def test_air_keeps_its_mixture_identity_and_uses_the_declared_oxygen_role():
    result = {"chemistry_plan": {"reagents": [{"name": "Molecular oxygen", "role": "terminal oxidant"}]}}
    stream = {"stream_label": "G", "contents": ["air"], "phase": "gas",
        "gas_active_species": "O2", "gas_reagent_mole_fraction": .21, "molar_equiv": 1.0}
    component = _stream_components(result, [stream])[0]
    assert component.name == "air"
    assert "gas mixture" in component.role and "terminal oxidant" in component.role
    assert component.molar_equiv == 1.0
    assert stream["gas_reagent_mole_fraction"] == .21
    assert _stream_components({"chemistry_plan": {}}, [stream])[0].role == "unknown"
