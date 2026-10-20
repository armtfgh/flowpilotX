# Continue after experimental feedback

The bundled demo is synthetic. For a real study, create your own campaign with reviewed inventory, constraints, candidate pool and observation contract before using its results for decisions.

1. Freeze a proposed round before outcomes arrive. Keep design IDs and independent repeat IDs distinct; record which proposals were actually executed.
2. Record actual measured settings, apparatus/calibration information, sample/fraction identity and deviations. Do not copy proposed or calculated values into actual fields as if measured.
3. Report starting material, target product, undesired products and other products separately, with analytical method, units, qualifiers and uncertainty where available. Preserve values above 100% and unknowns. An assay failure is not zero yield.
4. Import results against the saved `round_id` and `design_id`. Qualified observations require a documented analytical basis and explicit acceptance; retain incomplete and operationally failed runs.
5. Inspect the eligibility reasons. Batch evidence, incomparable conditions, missing measured controls, invalid assays and superseded results do not become numerical optimization feedback.
6. Update scientific questions with evidence, author and a reason. Set the next objective and number of conditions, review budgets and constraints, and freeze a new round. Earlier rounds remain unchanged.

For a correction, use a new observation ID with `supersedes` and `correction_reason`; preserve the earlier record. Fractions from one run and repeated analyses of one sample are not automatically independent experimental repeats.

The current optimizer ranks one declared objective and uses a transparent proximity/diversity policy. It does not fit a response surface or automatically enforce a newly mentioned byproduct threshold. Additional constraints must be explicitly implemented and tested before relying on them.

For fair algorithm comparisons, freeze information boundaries and use equal experiment/material budgets, common feasible hardware, consistent evaluation and independent repeats before seeing results. No benefit of historical data or model superiority is established by the included software examples.
