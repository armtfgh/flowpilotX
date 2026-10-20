# FlowPilot / FlowPilotX development instructions

Read README.md and docs/NEXT_THREAD.md before editing.

- Work in this existing repository. Keep the original FlowPilot source, retrieval software, engineering, diagrams, interfaces, generic FlowPilotX and study adapters. Do not replace the project with a smaller demonstration or create a nested release repository.
- Distinguish software from private data. Local raw records, inventories, populated indexes, campaign state, model calls and generated outputs are intentionally present but excluded from Git. Preserve their bytes and provenance. Cache cleanup must not remove functional code or experiment history.
- The original archive remains the pristine recovery source. Thirteen documented baseline changes cover publication portability, offline lexical retrieval, corpus filtering, optional-page handling and the frontend lockfile. Six supplemental source modules were recovered from the original project without editing it. Further baseline changes require original hashes and a recorded patch; do not silently rewrite the reference history.
- Keep the synthetic public demo independent of private study files. Do not describe fictional data as experiments or conditional proposals as validated procedures.
- Separate proposed settings, actual conditions, analytes, assumptions, calculated indices, source identity and missing values. Preserve reported values above 100 percent and unresolved discrepancies. Do not convert batch duration into flow residence time or pool incompatible catalyst/experiment modes.
- Preserve candidate pools, questions, rejection reasons, frozen rounds, configuration versions and append-only experimental corrections. Do not change evaluation definitions to favor an algorithm.
- The generic FlowPilotX GUI does not automatically invoke the original RAG/candidate-generation path. Document actual integration, not an intended architecture.
- Obtain explicit routing, information boundaries and campaign spending limits before live model or embedding work. Existing records and example configuration confer no new authorization. No hidden paid fallback, automatic corpus re-embedding, external messaging, uploads or equipment execution.
- Run focused offline checks. `python3 tools/run_public_tests.py` checks the data-free application. Original-component tests need optional dependencies and isolated data paths; never mutate a protected retrieval database during tests. Private study tests require their complete source bundle.
- Before publication, run `python3 tools/check_release.py` and inspect the staged diff. Exclude credentials, personal paths, laboratory endpoints and private study evidence from Git while retaining reusable source and assets. Do not assume ignored files already in Git history are hidden.
- No license has been granted automatically. Equipment-artwork rights and code licensing remain the owner's decision.
