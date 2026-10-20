from types import SimpleNamespace as NS

from flora_translate.schemas import BatchRecord, ChemistryPlan, ProcessStage, ReagentRole, StreamLogic
from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.scientific_evidence import source_context, scientific_class, assess_analogies


def example():
    batch = BatchRecord(reaction_description="DPDTC-mediated amide formation", reaction_time_h=1,
                        temperature_C=95, solvent="2-MeTHF", concentration_M=0.5,
                        raw_text="Acid, DPDTC and DMAP were charged in 2-MeTHF, heated and stirred for 30 min. "
                        "The vial cooled for 5-10 min. Benzylamine was added, then heated and stirred for another 30 min.")
    plan = ChemistryPlan(reaction_class="Amide formation", reagents=[
        ReagentRole(name=n, role=r) for n, r in [("Acid", "substrate"), ("DPDTC", "reagent"),
        ("DMAP", "catalyst"), ("Benzylamine", "reagent")]], stages=[
        ProcessStage(stage_number=1, feed_streams=[StreamLogic(stream_label="A", reagents=["Acid", "DPDTC", "DMAP", "Benzylamine"])]),
        ProcessStage(stage_number=2, feed_streams=[StreamLogic(stream_label="B", reagents=["Benzylamine"])])])
    return batch, plan


def test_source_stage_additions_and_holds_are_reproducible():
    b, p = example()
    assert source_context(b, p) == source_context(b, p)
    p, _ = reconcile_chemistry_plan(b, p, scientific=True)
    assert "Benzylamine" not in p.stages[0].feed_streams[0].reagents
    assert p.stages[1].feed_streams[0].reagents == ["Benzylamine"]
    assert p.stages[1].feed_streams[0].introduction_stage == 2
    assert [s.batch_time_h for s in p.stages] == [0.5, 0.5]
    assert not p.scientific_context["issues"]


def test_dpdtc_does_not_mean_palladium_coupling():
    b, p = example()
    assert scientific_class(b, p) == "amide_formation"
    b.reaction_description = "DPDTC activation"
    p.reaction_class = "thermal"
    assert scientific_class(b, p) != "cross-coupling"


def test_weak_unrelated_analogy_cannot_supply_kinetics():
    b, p = example()
    a = assess_analogies([{"final_score": 0.0444, "metadata": {"chemistry_class": "photochemical"}}], p)[0]
    assert a["retrieval_score"] == 0.0444
    assert not a["usable_as_kinetic_evidence"]


def test_ambiguous_hold_count_is_flagged_not_filled():
    b, p = example()
    b.raw_text = "Acid reacted. Benzylamine was added."
    assert source_context(b, p)["issues"]


def test_no_manufactured_kinetic_prediction():
    from flora_translate.design_calculator import DesignCalculator
    b, p = example()
    p, _ = reconcile_chemistry_plan(b, p, scientific=True)
    c = DesignCalculator().run(b, chemistry_plan=p)
    assert c.residence_time_min == 60
    assert c.rate_constant is None
    assert c.target_conversion is None
    assert c.intensification_factor is None
    assert c.kinetics_status == "uncharacterized"
    text = c.to_prompt_block()
    assert "Kinetics source:** unknown" in text
    assert "thermal safety uncharacterized" in text
    assert "class-level IF" not in text
    assert "95 % conversion" not in text
    assert c.steps[5].values["Da_mass"] is None
    assert c.thermal_safe is None
    assert c.heat_generation_W is None


def test_reversed_explicit_alias_and_sentence_punctuation():
    from flora_translate.scientific_evidence import identify
    b, p = example()
    p.reagents[1].name = "DPDTC (di(2-pyridyl) dithiocarbonate)"
    p.reagents[2].name = "DMAP (4-dimethylaminopyridine)"
    context = source_context(b, p)
    assert identify("DPDTC.", context["components"])["addition_stages"] == [1]
    assert identify("DMAP", context["components"])["addition_stages"] == [1]
    p, _ = reconcile_chemistry_plan(b, p, scientific=True)
    assert not p.scientific_context["issues"]


def test_cooling_duration_is_not_a_reaction_hold():
    b, p = example()
    b.raw_text = b.raw_text.replace("5-10 min", "5 min")
    assert len(source_context(b, p)["timed_holds"]) == 2
