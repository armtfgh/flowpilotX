from flora_translate.schemas import FlowProposal
from flora_translate.chemistry_agent import _normalize_plan_data
from flora_translate.schemas import ChemistryPlan
from flora_translate.translation_llm import TranslationLLM


def test_frontier_provider_wrappers_normalize_to_flow_proposal_schema():
    raw = {
        "residence_time_min": {"value": 25, "basis": "screening"},
        "flow_rate_mL_min": {"recommended": 0.1},
        "temperature_C": {"selected": 60},
        "concentration_M": {"nominal": 0.2},
        "BPR_bar": {"value": 5},
        "tubing_ID_mm": {"value": 1.0},
        "reactor_volume_mL": {"estimate": 2.5},
        "light_setup": {"name": "none", "reason": "thermal reaction"},
        "engine_validated": {"status": "passed"},
        "literature_analogies": [{"doi": "10.1000/example", "similarity": 0.8}],
        "streams": [
            {
                "stream_label": "A",
                "contents": ["substrate"],
                "concentration_M": {"value": 0.2},
                "molar_equiv": {"value": 1.0},
                "introduction_stage": {"value": 1},
            }
        ],
    }

    normalized = TranslationLLM._normalize_proposal_data(raw)
    proposal = FlowProposal.model_validate(normalized)

    assert proposal.residence_time_min == 25
    assert proposal.flow_rate_mL_min == 0.1
    assert proposal.engine_validated is True
    assert proposal.light_setup == "none"
    assert proposal.streams[0].concentration_M == 0.2
    assert proposal.literature_analogies[0].startswith("{")


def test_upstream_wrapped_values_normalize_to_chemistry_plan_schema():
    raw = {
        "reaction_name": {"name": "Hydrogenolysis"},
        "n_stages": {"value": 1},
        "oxygen_sensitive": {"value": False},
        "o2_is_reagent": {"value": False},
        "recommended_wavelength_nm": {"value": None},
        "stream_logic": [
            {
                "stream_label": "A",
                "reagents": ["substrate"],
                "molar_equiv": {"value": 1.0},
                "concentration_M": {"value": 0.1},
            }
        ],
        "stages": [
            {
                "stage_number": {"value": 1},
                "requires_light": {"value": False},
                "temperature_C": {"selected": 60},
                "feed_streams": [],
            }
        ],
    }

    plan = ChemistryPlan.model_validate(_normalize_plan_data(raw))

    assert plan.reaction_name == "Hydrogenolysis"
    assert plan.n_stages == 1
    assert plan.stages[0].temperature_C == 60
    assert plan.stream_logic[0].concentration_M == 0.1
