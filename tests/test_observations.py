import unittest

from khu_sulfoxidation.observations import AnalyticalOutcome, ExperimentObservation, results_template
from khu_sulfoxidation.feedback import qualitative_feedback
from khu_sulfoxidation.evidence import assess_objective


class ObservationTests(unittest.TestCase):
    def test_missing_is_unknown_and_no_mass_closure(self):
        outcome = AnalyticalOutcome(residual_sulfide_pct=0, sulfoxide_pct=101)
        self.assertEqual(outcome.sulfoxide_pct, 101)
        self.assertIsNone(outcome.sulfone_pct)
        self.assertIsNone(outcome.component_sum_pct)
        self.assertIsNone(outcome.two_product_selectivity_pct)
        self.assertIn("component_percentage_outside_0_100", outcome.flags)
        self.assertIsNone(assess_objective(outcome, minimum_sulfoxide_pct=90, maximum_sulfone_pct=5)["meets_reported_endpoints"])

    def test_actual_and_proposed_and_operational_unknowns_separate(self):
        proposed = {"liquid_flow_mL_min": 0.1}
        result = results_template("test-01", proposed)
        proposed["liquid_flow_mL_min"] = 0.2
        self.assertEqual(result["proposed_conditions"]["liquid_flow_mL_min"], 0.1)
        self.assertIsNone(result["actual_conditions"])
        self.assertIsNone(result["operational_observations"]["clogging_observed"])
        self.assertIsNone(result["operational_observations"]["post_collection_oxidation_observed"])
        self.assertIsNone(result["analytical_method"])

    def test_batch_time_not_residence_time(self):
        with self.assertRaises(ValueError):
            ExperimentObservation("batch-1", "batch", actual_conditions={"residence_time_min": 10})
        result = ExperimentObservation("batch-1", "batch", actual_conditions={"batch_duration_h": 10})
        self.assertEqual(result.actual_conditions["batch_duration_h"], 10)

    def test_sulfone_rich_feedback_can_hypothesize_shorter_time(self):
        result = ExperimentObservation("flow-1", "flow", actual_conditions={"liquid_only_space_time_min": 5},
                                       outcomes=AnalyticalOutcome(0, 40, 58, 0, 0))
        feedback = qualitative_feedback(result, sulfone_limit_pct=5, residual_sulfide_review_pct=10)
        self.assertIn("sulfone_above_declared_limit", feedback["diagnoses"])
        self.assertIn("shorter exposure", feedback["hypotheses"][0]["hypothesis"])
        self.assertNotIn("residual_sulfide_above_declared_review_level", feedback["diagnoses"])
        self.assertIsNone(feedback["fitted_kinetics"])
        self.assertEqual(feedback["parameter_changes"], {})

    def test_incomplete_reaction_different_from_sulfone_rich(self):
        result = ExperimentObservation("flow-2", "flow", outcomes=AnalyticalOutcome(80, 1, 0))
        feedback = qualitative_feedback(result, sulfone_limit_pct=5, residual_sulfide_review_pct=10)
        self.assertIn("residual_sulfide_above_declared_review_level", feedback["diagnoses"])
        self.assertNotIn("sulfone_above_declared_limit", feedback["diagnoses"])
        self.assertIn("actual_conditions_missing", feedback["diagnoses"])

    def test_nonfinite_values_not_serialized_as_measurements(self):
        for value in (float("nan"), float("inf"), True, "98"):
            with self.assertRaises(ValueError):
                AnalyticalOutcome(sulfoxide_pct=value)


if __name__ == "__main__":
    unittest.main()
