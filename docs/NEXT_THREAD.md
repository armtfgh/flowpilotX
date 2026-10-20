# Development handoff — 16 October 2026

The active repository is the existing `FlowPilotX` directory, formerly named `flowpilot-khu-sulfoxidation`. There is no nested GitHub delivery folder. The complete software has been restored after an overly broad cleanup: original FlowPilot modules and assets, generic FlowPilotX, the KHU adapter, scripts and tests. The pristine private archive remains available to the owner; do not delete it.

## Current software

- `baseline/flowpilot/`: RAG, embedding and vector-store code, chemistry interpretation, specialist councils, equipment/engineering calculations, topology renderer/icons, Streamlit interface and FastAPI/React interface.
- `src/flowpilotx/`: question-led campaigns, N-condition selection, repeats, evidence and inventory contracts, immutable rounds, qualified feedback, bounded local-model gateway and browser GUI.
- `src/khu_sulfoxidation/`: historical-source reconciliation, coupled flow arithmetic, study candidates, model-review handling, diagrams and reports.

The generic GUI does not automatically call the original RAG/translation path to generate new pools. Its current numerical selector uses coverage/proximity/diversity heuristics, not fitted kinetics or Bayesian optimization. Source availability does not imply every original interface or historical benchmark has been validated in a clean environment. Missing mining/extraction/figure modules were subsequently recovered as six supplemental source files from the original project, which was read but never modified. Their hashes and eight private figure assets are recorded separately from the original 820-file handoff.

## Local research versus public Git

Private inputs, inventories, populated literature/index data, archived calls and saved campaigns have been restored locally and are ignored by Git. A public clone starts with the synthetic demo and needs owner-provided data for retrieval or study-specific work. Do not copy unpublished details into public examples or documentation.

The owner's local `docs/NEXT_EXPERIMENT_GUIDE.md` contains the study continuation workflow. Existing source records and frozen model proposals remain evidence, not new spending authorization or laboratory release. No additional model calls were made during restoration.

The archive retains all 820 original protected handoff files unchanged. The active protected set now has 807 unchanged files and 13 explicitly documented baseline patches. The initial five publication patches configure loopback endpoints and generic examples; later patches repair standalone offline lexical retrieval, corpus filtering, optional-page handling and a frontend dependency. Original copies, original/updated hashes and unified diffs are recorded in the ignored local `provenance/baseline_*patches_*/manifest.json` files. Supplemental recovered modules/assets have a separate manifest under `provenance/supplemental_flowpilot_modules_2026-10-16_v1/`. Do not claim active baseline bytes are all identical to the pristine snapshot.

Reviewed slide facts, laboratory correspondence and source-specific profile details now load from ignored `data/private/khu_source_context_v1.json`, with a pinned source hash and an explicit error if missing. Original adapter files and exact diffs are kept in the local restoration provenance records. Tests confirmed unchanged scientific content; newly calculated profile provenance records the changed source hash. Frozen historical campaigns were not rewritten. A Decimal sum also fixes a Python-version-dependent preflight currency display; routing, limits and authorization are unchanged.

## Start and check

```bash
python3 scripts/run_flowpilotx_platform.py --profile demo --port 8788
python3 tools/run_public_tests.py
python3 tools/check_release.py
```

Use `--profile khu` only with the complete private source bundle. Original FlowPilot launch commands and dependencies are in [README](../README.md) and [Configuration](CONFIGURATION.md). Import or test the original vector store only against isolated writable copies or synthetic stores; opening a database can create or update local state.

Restoration verification: the public runner passes 48 tests. The complete local adapter/application suite runs 249 tests: 248 pass and one is skipped. Separately, 111 selected original component tests and four tests using unpopulated RAG stores pass. An additional 18 original checks could not run successfully because pre-existing external fixture files were absent; these are not a completed full baseline suite. None establishes experimental validation or an end-to-end paid campaign.

With the complete private bundle, run the local adapter/application suite from the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

The restored KHU GUI opens both saved campaigns and its default six unique conditions plus one separate repeat. Its live model routes remain disabled. No calls were used for this verification.

Continue integration through explicit adapters and traceable records. Do not reset model ledgers, substitute a new demo for a saved campaign, or automatically rebuild the corpus. License and equipment-artwork rights still require review before public redistribution; see [Publication](PUBLICATION.md).

Actual runtime verification and remaining limitations are in [Runtime status](RUNTIME_STATUS.md). The reusable local `.venv` and built React assets are present and Git-ignored. Verification servers were stopped after testing; no live model or re-embedding calls were made.
