from flora_translate.intake_agent import IntakeAgent, _gas_introduction_stage, _conditional_answer_resolved
import pytest


@pytest.mark.parametrize("answer,expected", [
    ("Oxygen should first be introduced as the Stage 1 effluent enters the oxidation stage.", 2),
    ("Add gas as the Stage 2 effluent enters Stage 3", 3),
    ("Stage 2", 2), (2, 2),
    ("Introduce oxygen at the inlet of Stage 1, into Reactor 1.", 1),
    ("Stage 1 or Stage 2", None), ("Not Stage 1", None),
])
def test_gas_destination_not_source(answer, expected):
    assert _gas_introduction_stage(answer) == expected


def test_delegated_oxygen_feed_is_not_a_fabricated_equivalent():
    agent = IntakeAgent()
    package = agent.analyze("Two-stage oxidation under blue light and oxygen.", use_llm=False,
        answers=[{"question_id": "Q-GAS-002", "answer": "No fixed value. Please determine and justify an initial oxygen feed."},
                 {"question_id": "Q-GAS-003", "answer": "Oxygen should first be introduced as the Stage 1 effluent enters the oxidation stage."}])
    gas = package.engineering_requirements["gas"]
    assert gas["introduction_stage"] == 2
    assert "target_equiv_inlet_stp" not in gas
    assert gas["feed_selection_mode"] == "agent_to_propose"
    assert _conditional_answer_resolved(package, "Q-GAS-002")
    again = agent.analyze(existing_package=package, use_llm=False)
    assert again.engineering_requirements == package.engineering_requirements
    updated = agent.analyze(existing_package=package, use_llm=False, answers=[
        {"question_id": "Q-GAS-002", "answer": "2.0 equiv"},
        {"question_id": "Q-GAS-003", "answer": "Stage 1 or Stage 2"}])
    assert updated.engineering_requirements["gas"]["target_equiv_inlet_stp"] == 2.0
    assert "feed_selection_mode" not in updated.engineering_requirements["gas"]
    assert "introduction_stage" not in updated.engineering_requirements["gas"]


def test_delegated_feed_preserves_model_proposal_without_chemist_authority():
    from flora_translate.schemas import DesignInputPackage, ChemistryPlan, StreamLogic
    from flora_translate.intake_agent import apply_intake_requirements_to_chemistry_plan
    package = DesignInputPackage(engineering_requirements={"gas": {
        "species": "O2", "reagent_mole_fraction": 1.0,
        "feed_selection_mode": "agent_to_propose", "introduction_stage": 2}})
    plan = ChemistryPlan(stream_logic=[StreamLogic(stream_label="G", reagents=["O2"], phase="gas", molar_equiv=2.5)])
    updated, decisions = apply_intake_requirements_to_chemistry_plan(plan, package)
    assert updated.stream_logic[0].molar_equiv == 2.5
    assert updated.stream_logic[0].requirement_authority == "model_inference"
    assert decisions[0]["target_equiv_inlet_stp"] is None


def test_revised_gas_answer_replaces_effective_stage_without_erasing_history():
    agent = IntakeAgent()
    old = agent.analyze("Two-stage photochemical oxidation with oxygen.", use_llm=False,
        answers=[{"question_id": "Q-GAS-003", "answer": "Stage 2"}])
    revised = agent.analyze(existing_package=old, use_llm=False,
        answers=[{"question_id": "Q-GAS-003", "answer": "Introduce oxygen at the inlet of Stage 1, into Reactor 1."}])
    assert old.engineering_requirements["gas"]["introduction_stage"] == 2
    assert revised.engineering_requirements["gas"]["introduction_stage"] == 1
    assert revised.answer_map()["Q-GAS-003"].answer.startswith("Introduce oxygen")
    assert len([a for a in revised.answers if a.question_id == "Q-GAS-003"]) == 2
    again = agent.analyze(existing_package=revised, use_llm=False)
    assert again.engineering_requirements["gas"]["introduction_stage"] == 1
    from flora_translate.intake_agent import intake_context_block
    prompt = intake_context_block(revised)
    assert '"answer": "Stage 2"' not in prompt
    assert '"answer": "Introduce oxygen at the inlet of Stage 1, into Reactor 1."' in prompt
