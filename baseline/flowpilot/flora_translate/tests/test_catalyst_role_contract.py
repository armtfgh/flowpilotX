"""Stationary-phase assignment must not depend on exact model vocabulary."""

import pytest

from flora_translate.component_identity import has_catalytic_role
from flora_translate.design_realizer import _is_stationary_catalyst
from flora_translate.final_design_contract import _resolved_stationary_components


@pytest.mark.parametrize("role", [
    "heterogeneous catalyst", "heterogeneous precatalyst",
    "immobilized pre-catalyst", "immobilized pre catalyst",
    "fixed-bed photocatalyst", "fixed-bed electrocatalyst",
    "stationary catalytic material",
])
def test_supported_catalytic_roles_agree_across_realization_and_final_gate(role):
    assert has_catalytic_role(role)
    assert _is_stationary_catalyst(role=role)
    resolved = _resolved_stationary_components(
        {"reactor_type": "packed-bed"},
        {"reagents": [{"name": "material A", "role": role}]},
    )
    assert [item["name"] for item in resolved] == ["material A"]


@pytest.mark.parametrize("role", [
    "soluble catalyst", "homogeneous precatalyst", "photocatalyst in solution",
    "substrate", "stoichiometric reagent", "heterogeneous adsorbent",
])
def test_catalysis_alone_does_not_authorize_stationary_placement(role):
    assert not _is_stationary_catalyst(role=role)
    assert not _resolved_stationary_components(
        {"reactor_type": "packed-bed"},
        {"reagents": [{"name": "material A", "role": role}]},
    )


def test_stationary_role_requires_a_packed_bed_not_an_ordinary_coil():
    assert not _resolved_stationary_components(
        {"reactor_type": "coil"},
        {"reagents": [{"name": "material A", "role": "heterogeneous precatalyst"}]},
    )


@pytest.mark.parametrize("loading,expected", [
    ("Flow: use the supplied 3 mL prepacked bed; mass must be verified", True),
    ("Flow: use the supplied packed bed", True),
    ("Installed cartridge; verify actual solid mass", True),
    ("Use the supplied 3 mL packed reactor; actual solid mass unresolved", True),
    ("Retained in a packed column", True),
    ("", False),
    ("0.5 g in the liquid feed", False),
])
def test_supported_precursor_requires_explicit_fixed_phase_loading(loading, expected):
    role = "heterogeneous reaction precursor"
    assert _is_stationary_catalyst(role=role, declared_loading=loading) is expected
    resolved = _resolved_stationary_components(
        {"reactor_type": "packed-bed"},
        {"reagents": [{"name": "material A", "role": role, "equiv_or_loading": loading}]},
    )
    assert bool(resolved) is expected


def test_soluble_precursor_passing_through_bed_remains_a_feed_component():
    assert not _is_stationary_catalyst(
        role="soluble reaction precursor", declared_loading="0.1 M; feed to packed bed",
    )


@pytest.mark.parametrize("role", ["Supported hydrogenolysis material", "Supported copper source for cycloaddition", "Immobilized scavenging reagent", "Heterogeneous adsorbent"])
def test_explicit_fixed_phase_does_not_require_catalyst_vocabulary(role):
    assert _is_stationary_catalyst(role=role, declared_loading="Flow: use the supplied packed-bed module; bed mass unresolved")
    assert not _is_stationary_catalyst(role=role, declared_loading="Feed a slurry through the packed bed")


def test_historical_bed_does_not_override_current_liquid_assignment():
    assert not _is_stationary_catalyst(role="Supported metal source", declared_loading="Batch: packed bed. Flow: dissolve in the liquid feed.")
