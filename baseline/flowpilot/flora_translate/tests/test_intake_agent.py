import itertools

import pytest

from flora_translate.design_calculator import DesignCalculator
from flora_translate.intake_agent import (
    IntakeAgent,
    QUESTION_BANK,
    apply_intake_requirements_to_chemistry_plan,
    intake_context_block,
)
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    DesignInputPackage,
    FlowProposal,
    IntakeAnswer,
    ProcessStage,
    StreamAssignment,
    StreamLogic,
)


def test_design_input_package_roundtrip():
    package = DesignInputPackage(
        raw_protocol="Substrate to product in MeCN, 0.1 M, 25 C, 4 h, 80% yield.",
        objective="first safe screening design",
        historical_data="none",
        inventory_constraints={"reactors": []},
        hypotheses=["kinetics may be slow"],
        operating_limits={"max_pressure_bar": 8},
        ready_for_design=True,
    )

    restored = DesignInputPackage.model_validate_json(package.model_dump_json())

    assert restored.schema_version == "flowpilot_intake_v1.1"
    assert restored.question_bank_version == "flowpilot_questions_v1.1"
    assert restored.objective == "first safe screening design"
    assert restored.hypotheses == ["kinetics may be slow"]


def test_intake_agent_generates_stable_fixed_question_ids_without_llm():
    agent = IntakeAgent()
    protocol = "A to B in DMSO, 0.5 M, 40 C, 15 h, blue LED, O2."

    first = agent.analyze(protocol, use_llm=False)
    second = agent.analyze(protocol, use_llm=False)

    assert first.missing_question_ids == second.missing_question_ids
    assert set(first.missing_question_ids) <= set(QUESTION_BANK)
    assert first.missing_question_ids == [
        "Q-OBJ-001",
        "Q-CHEM-001",
        "Q-HIST-001",
        "Q-INV-001",
        "Q-CONSTR-001",
        "Q-HYP-001",
        "Q-PHOTO-001",
    ]
    assert not first.ready_for_design


def test_intake_agent_explicit_deterministic_mode_does_not_call_llm(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called by default intake analysis")

    import flora_translate.intake_agent as intake_agent

    monkeypatch.setattr(intake_agent, "call_model_text", fail_if_called)

    package = IntakeAgent().analyze("A to B in DMSO, 0.5 M, 40 C, 15 h.", use_llm=False)

    assert package.extracted_batch_fields["concentration_M"] == 0.5
    assert "Q-OBJ-001" in package.missing_question_ids


def test_intake_agent_blocks_until_required_sections_answered_or_unavailable():
    agent = IntakeAgent()
    answers = [
        IntakeAnswer(question_id="Q-OBJ-001", answer="next longer screening design"),
        IntakeAnswer(
            question_id="Q-CHEM-001",
            answer="Oxidation of A to B.",
        ),
        IntakeAnswer(question_id="Q-HIST-001", answer="Entry 1: tau 35 min, product 10%"),
        IntakeAnswer(question_id="Q-INV-001", status="unavailable"),
        IntakeAnswer(question_id="Q-CONSTR-001", status="unavailable"),
        IntakeAnswer(question_id="Q-HYP-001", answer="Longer residence time may be needed."),
    ]

    package = agent.analyze(
        "A to B in DMSO, 0.5 M, 40 C, 15 h.",
        answers=answers,
        use_llm=False,
    )

    assert package.ready_for_design
    assert package.missing_question_ids == []
    assert package.objective == "next longer screening design"
    assert package.inventory_constraints is None
    assert package.hypotheses == ["Longer residence time may be needed."]


def test_intake_requires_chemist_identity_when_protocol_family_is_ambiguous():
    agent = IntakeAgent()
    package = agent.analyze(
        "Compound A was treated with reagent B and afforded compound C.",
        use_llm=False,
    )

    assert "Q-CHEM-001" in package.missing_question_ids
    assert not package.chemistry_identity_confirmation


def test_protocol_stated_identity_is_frozen_without_extra_llm_authority():
    agent = IntakeAgent()
    package = agent.analyze(
        "Hydrogenolysis/debenzylation of protected amine A afforded amine B.",
        use_llm=False,
    )

    assert "Q-CHEM-001" not in package.missing_question_ids
    assert package.chemistry_identity_confirmation == {
        "transformation_family": "hydrogenolysis",
        "confirmed": True,
        "source": "protocol_fact",
    }


def test_plain_language_chemist_identity_closes_the_required_question():
    agent = IntakeAgent()
    initial = agent.analyze(
        "Compound A was treated with reagent B and afforded compound C.",
        use_llm=False,
    )
    assert "Q-CHEM-001" in initial.missing_question_ids

    package = agent.analyze(
        initial.raw_protocol,
        existing_package=initial,
        answers=[
            IntakeAnswer(
                question_id="Q-CHEM-001",
                answer="Ring closure of precursor A gives cyclic product C.",
            )
        ],
        use_llm=False,
    )

    assert "Q-CHEM-001" not in package.missing_question_ids
    assert package.chemistry_identity_confirmation["confirmed"] is True
    assert package.chemistry_identity_confirmation["chemist_description"].startswith(
        "Ring closure"
    )


def test_required_chemist_identity_cannot_be_marked_unavailable():
    package = IntakeAgent().analyze(
        "Compound A was treated with reagent B and afforded compound C.",
        answers=[IntakeAnswer(question_id="Q-CHEM-001", status="unavailable")],
        use_llm=False,
    )

    assert "Q-CHEM-001" in package.missing_question_ids
    assert not package.ready_for_design


def test_intake_context_block_preserves_authority_labels():
    package = DesignInputPackage(
        raw_protocol="A to B.",
        objective="screen",
        historical_data="Measured: 10% product at 35 min.",
        inventory_constraints={"BPR_available": [6]},
        hypotheses=["photon limitation"],
        operating_limits="max pressure 8 bar",
        ready_for_design=True,
    )

    block = intake_context_block(package)

    assert "measured evidence > hard constraints > protocol facts" in block
    assert "Measured: 10% product" in block
    assert "photon limitation" in block


def test_intake_context_compacts_inventory_answer_without_losing_capabilities():
    equipment = {
        "equipment_id": "KHU-REACTOR-10ML",
        "name": "KHU 10 mL PFA coil",
        "volume_mL": 10,
        "ID_mm": 1.0,
        "max_pressure_bar": 8,
        "notes": "x" * 40000,
    }
    package = DesignInputPackage(
        raw_protocol="A to B.",
        objective="screen",
        inventory_constraints={"reactors": [equipment]},
        answers=[
            IntakeAnswer(question_id="Q-INV-001", answer={"reactors": [equipment]})
        ],
        ready_for_design=True,
    )

    block = intake_context_block(package)

    assert "KHU-REACTOR-10ML" in block
    assert '"volume_mL": 10' in block
    assert len(block) < 5000


def test_fallback_extract_preserves_explicit_photocatalyst_and_solvent():
    protocol = (
        "Photocatalyst: Ir(dF(CF3)ppy)2(dtbpy)PF6, 0.5 mol%. "
        "Solvent: EtOH/pH 9 aqueous buffer, 5:1 v/v. Concentration: 0.10 M. "
        "Irradiate with a 452 nm LED under argon."
    )

    package = IntakeAgent().analyze(protocol, use_llm=False)

    assert package.extracted_batch_fields["photocatalyst"] == "Ir(dF(CF3)ppy)2(dtbpy)PF6"
    assert package.extracted_batch_fields["catalyst_loading_mol_pct"] == 0.5
    assert package.extracted_batch_fields["solvent"] == "EtOH/pH 9 aqueous buffer, 5:1 v/v"
    assert package.extracted_batch_fields["atmosphere"] == "argon"


ADAPTIVE_PROTOCOL = (
    "A two-stage photochemical oxidation is performed with oxygen gas under "
    "3 bar using a blue LED. Compound A is converted to product B."
)


def _required_answers(gas_equiv: float) -> list[IntakeAnswer]:
    return [
        IntakeAnswer(question_id="Q-OBJ-001", answer="first executable screen"),
        IntakeAnswer(question_id="Q-CHEM-001", answer="Two-stage oxidation of A to B."),
        IntakeAnswer(question_id="Q-HIST-001", status="unavailable"),
        IntakeAnswer(question_id="Q-INV-001", status="unavailable"),
        IntakeAnswer(question_id="Q-CONSTR-001", status="unavailable"),
        IntakeAnswer(question_id="Q-HYP-001", status="unavailable"),
        IntakeAnswer(question_id="Q-GAS-002", answer=f"{gas_equiv} equiv"),
        IntakeAnswer(question_id="Q-GAS-003", answer="Stage 2"),
        IntakeAnswer(question_id="Q-PHOTO-001", answer="450 nm"),
        IntakeAnswer(
            question_id="Q-MULTI-001",
            answer="Stage 1: photochemical formation. Stage 2: oxygen oxidation.",
        ),
    ]


def test_conditional_question_set_is_identical_over_repeated_analysis():
    agent = IntakeAgent()
    packages = [agent.analyze(ADAPTIVE_PROTOCOL, use_llm=False) for _ in range(20)]

    assert len({tuple(pkg.active_question_ids) for pkg in packages}) == 1
    assert len({tuple(pkg.missing_question_ids) for pkg in packages}) == 1
    assert len({pkg.question_set_hash for pkg in packages}) == 1
    assert packages[0].active_question_ids[-4:] == [
        "Q-GAS-002", "Q-GAS-003", "Q-PHOTO-001", "Q-MULTI-001"
    ]


def test_question_selection_does_not_depend_on_varying_llm_extraction(monkeypatch):
    agent = IntakeAgent()
    variants = itertools.cycle([
        {"reaction_description": "model extraction A", "temperature_C": 20},
        {"reaction_description": "model extraction B", "temperature_C": 80},
    ])
    monkeypatch.setattr(agent, "_extract_batch_fields", lambda *args, **kwargs: next(variants))

    packages = [agent.analyze(ADAPTIVE_PROTOCOL, use_llm=True) for _ in range(8)]

    assert len({tuple(pkg.active_question_ids) for pkg in packages}) == 1
    assert len({pkg.question_set_hash for pkg in packages}) == 1


def test_gas_equivalent_answer_changes_stp_flow_and_primary_residence_geometry():
    agent = IntakeAgent()
    packages = [
        agent.analyze(
            ADAPTIVE_PROTOCOL,
            answers=_required_answers(equiv),
            use_llm=False,
        )
        for equiv in (1.0, 2.0)
    ]
    base_plan = ChemistryPlan(
        reaction_class="photochemical oxidation",
        mechanism_type="gas-liquid oxidation",
        n_stages=2,
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["A"], concentration_M=0.5),
            StreamLogic(stream_label="G", reagents=["O2"], phase="gas"),
        ],
        stages=[
            ProcessStage(stage_number=1, stage_name="formation", requires_light=True),
            ProcessStage(stage_number=2, stage_name="oxidation"),
        ],
    )
    batch = BatchRecord(
        reaction_description=ADAPTIVE_PROTOCOL + " Stage 2 is open to air.",
        raw_text=ADAPTIVE_PROTOCOL + " Stage 2 is open to air.",
        concentration_M=0.5,
        temperature_C=40,
        reaction_time_h=10,
        atmosphere="O2",
    )
    results = []
    for package in packages:
        assert package.ready_for_design
        plan, _ = apply_intake_requirements_to_chemistry_plan(base_plan, package)
        proposal = FlowProposal(
            residence_time_min=60,
            residence_time_basis="inlet/STP apparent residence time",
            flow_rate_mL_min=0.02,
            concentration_M=0.5,
            temperature_C=40,
            BPR_bar=6,
            tubing_ID_mm=1.0,
            streams=[
                StreamAssignment(
                    stream_label="A", contents=["A"], concentration_M=0.5,
                    flow_rate_mL_min=0.02,
                ),
                StreamAssignment(stream_label="G", contents=["O2"], phase="gas"),
            ],
        )
        results.append(DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal))

    one, two = results
    assert one.residence_time_basis == "inlet/STP apparent residence time"
    assert two.residence_time_basis == "inlet/STP apparent residence time"
    assert two.gas_flow_sccm == pytest.approx(2 * one.gas_flow_sccm, rel=1e-3)
    assert two.gas_equiv_supplied == pytest.approx(2.0, rel=1e-3)
    assert two.reactor_volume_mL > one.reactor_volume_mL


def test_explicit_gas_composition_answer_propagates_to_chemistry_plan():
    protocol = "A gas-liquid oxidation uses a reagent gas under 3 bar."
    package = IntakeAgent().analyze(
        protocol,
        answers=[
            *_required_answers(2.0)[:6],
            IntakeAnswer(question_id="Q-GAS-001", answer="O2, 0.50 mole fraction"),
            IntakeAnswer(question_id="Q-GAS-002", answer="2 equiv"),
        ],
        use_llm=False,
    )
    plan, _ = apply_intake_requirements_to_chemistry_plan(ChemistryPlan(), package)
    gas = next(feed for feed in plan.stream_logic if feed.phase == "gas")

    assert gas.gas_reagent_mole_fraction == pytest.approx(0.5)
    assert gas.molar_equiv == pytest.approx(2.0)


def test_oxygen_free_first_stage_does_not_steal_second_stage_air_feed():
    protocol = (
        "Step 1: Maintain strictly oxygen-free conditions under argon. "
        "Step 2: Open to air and continue irradiation for aerobic oxidation."
    )

    package = IntakeAgent().analyze(protocol, use_llm=False)

    assert package.engineering_requirements["gas"]["species"] == "air"
    assert package.engineering_requirements["gas"]["introduction_stage"] == 2
    assert "Q-GAS-003" not in package.active_question_ids


@pytest.mark.parametrize('answer,species,fraction', [
    ('Pure oxygen (O2), 100 mol% O2. Require pure oxygen at Stage 2; do not use air.', 'O2', 1.0),
    ('Use pure oxygen, not air.', 'O2', 1.0),
    ('Use oxygen instead of compressed air.', 'O2', 1.0),
    ('Use air rather than pure oxygen.', 'air', .21),
    ('O2, 50 mol% in nitrogen.', 'O2', .5),
    ('Air (21% oxygen).', 'air', .21),
])
def test_explicit_gas_answer_excludes_negated_alternatives(answer, species, fraction):
    package = IntakeAgent().analyze(
        'Step 1: under argon. Step 2: open to air for oxidation.',
        answers=[IntakeAnswer(question_id='Q-GAS-001', answer=answer)],
        use_llm=False,
    )
    gas = package.engineering_requirements['gas']
    assert gas['identity_source'] == 'chemist_answer'
    assert gas['species'] == species
    assert gas['reagent_mole_fraction'] == pytest.approx(fraction)
