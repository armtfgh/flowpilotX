from components.intake_wizard import _apply_inventory_profile
from flora_translate.intake_agent import IntakeAgent
from flora_translate.inventory_profiles import (
    EquipmentCapability,
    inventory_profile_from_payload,
)


def test_profile_fills_fixed_inventory_and_constraint_answers():
    profile = inventory_profile_from_payload(
        {
            "pumps": [],
            "tubing": [],
            "BPR_available": [3],
            "light_sources": [],
            "gas_hardware": [],
            "reactors": [
                {
                    "type": "coil",
                    "material": "FEP",
                    "volume_mL": 10,
                    "ID_mm": 1.0,
                }
            ],
        },
        default_name="Design inventory",
    )
    profile.equipment_capabilities["inline_degassing"] = EquipmentCapability(
        available=False,
        service_status="unavailable",
        allowed_alternatives=["offline argon sparging"],
    )
    agent = IntakeAgent()
    package = agent.analyze(
        "Photochemical batch protocol at 25 C for 4 hours.",
        answers=[
            {
                "question_id": "Q-OBJ-001",
                "answer": "Create a first screening design.",
            }
        ],
        use_llm=False,
    )

    updated = _apply_inventory_profile(
        agent,
        package.raw_protocol,
        package,
        profile,
    )

    answers = updated.answer_map()
    assert answers["Q-INV-001"].source == "inventory_profile"
    assert answers["Q-CONSTR-001"].source == "inventory_profile"
    assert updated.inventory_constraints["reactors"][0]["volume_mL"] == 10
    assert updated.operating_limits["inline_degasser_available"] is False
    assert updated.inventory_profile_snapshot["name"] == "Design inventory"
