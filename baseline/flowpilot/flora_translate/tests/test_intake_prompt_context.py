from flora_translate.prompt_builder import TranslationPromptBuilder
from flora_translate.schemas import BatchRecord, ChemistryPlan, DesignInputPackage


def _package():
    return DesignInputPackage(
        raw_protocol="A to B in DMSO.",
        extracted_batch_fields={
            "reaction_description": "A to B in DMSO",
            "solvent": "DMSO",
            "temperature_C": 40,
        },
        objective="next longer screening design",
        historical_data="Measured evidence: KRICT gave 10-12% product at 34-38 min.",
        inventory_constraints={"reactors": [{"volume_mL": 10, "ID_mm": 1.0}]},
        hypotheses=["oxygen mass transfer may be limiting"],
        operating_limits="max pressure 8 bar",
        ready_for_design=True,
    )


def test_translation_prompt_includes_authority_labeled_intake_context():
    batch = BatchRecord(
        reaction_description="A to B in DMSO",
        solvent="DMSO",
        temperature_C=40,
    )
    plan = ChemistryPlan(reaction_class="photochemical oxidation")

    _, user_prompt = TranslationPromptBuilder().build(
        batch,
        analogies=[],
        chemistry_plan=plan,
        calculations=None,
        inventory=None,
        intake_package=_package(),
    )

    assert "FlowPilot Intake Context - authority labeled" in user_prompt
    assert "measured evidence > hard constraints > protocol facts" in user_prompt
    assert "KRICT gave 10-12% product" in user_prompt
    assert "oxygen mass transfer may be limiting" in user_prompt
    assert "Measured evidence in the intake context overrides" in user_prompt
