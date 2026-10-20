from flora_translate.executable_artifacts import _stream_components


def test_zero_equivalent_is_unresolved_instead_of_schema_crash():
    components = _stream_components(
        {"chemistry_plan": {"reagents": []}},
        [
            {
                "stream_label": "A",
                "phase": "liquid",
                "contents": ["unquantified additive (0 equiv)"],
            }
        ],
    )

    assert len(components) == 1
    assert components[0].molar_equiv is None
    assert components[0].quantified is False
