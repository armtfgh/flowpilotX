# Runtime verification — 16 October 2026

The restored interfaces and local retrieval were executed against isolated copies of the private data. These checks go beyond source presence and mocked component tests. No model generation, new embeddings, external literature searches or provider extraction calls were made.

| Component | Actual verification | Limit |
|---|---|---|
| Original React/FastAPI application | Production build; 11 endpoint checks; 13 browser checks for topology, design tabs, inventory and deterministic intake; original webapp tests: 31 passed, 7 skipped | Seven tests require separate saved-model archives; no fresh model-generated design was run |
| FlowPilotX GUI | Both saved study campaigns, all 12 design diagrams, seven navigation views, a new offline six-condition round plus repeat, seven-row CSV export | Tests use copied campaign state; original rounds remain unchanged |
| Original Streamlit application | 13 executed states, saved inventory loading, protocol intake, corpus and handbook display, Figure Studio gallery and editor | Live mining/extraction and model-backed design are not exercised |
| Populated RAG | Chroma opens 437 records and 115 batch–flow pairs; lexical queries return loadable source records; existing-vector self-search returns its source at distance zero | No new semantic query embedding or scientific relevance benchmark |
| Retrieval regression | Six tests pass; standalone lexical retrieval no longer requests an LLM summary when no chemistry plan is provided | Semantic retrieval can still invoke configured model providers |
| Public source package | 48 synthetic tests pass without private inputs | Does not validate laboratory performance |

## Repairs made

- Installed missing FastAPI runtime dependencies in the ignored local `.venv`; tested compatible backend/Chroma versions are recorded in `requirements-runtime-constraints.txt`.
- Built the original React frontend and updated one source-map dependency in its lockfile.
- Corrected the corpus browser to retain unknown publication years by default. It now displays the 437 indexed records instead of hiding them.
- Made lexical retrieval without a chemistry plan use only local query text.
- Recovered six source files for literature mining, PDF extraction and Figure Studio from the untouched original project. Restored figure assets remain private. Pages explain missing optional inputs instead of crashing.
- Changed configured-model labels so they do not imply a verified live connection.

Every protected-source change has original backups, hashes and a recorded patch. The original 820-file archive remains intact. The active protected set contains 807 unchanged files and 13 documented patches; six supplemental source files and eight private figure assets have their own provenance manifest.

## Remaining boundaries

The literature directory contains 492 JSON files: 437 normalize successfully, 54 fail the existing schema, and one is a cost log. The failed records were retained unchanged; they are not represented in the searchable index. Missing classification values and qualified/range yields need a deliberate, provenance-preserving schema review before expanding the corpus. No re-embedding was attempted.

The sample sulfoxidation query returned broad photochemistry analogies rather than close reaction matches. They are marked ineligible for quantitative transfer. Successful retrieval plumbing is not evidence that the corpus is sufficient for this chemistry.

Original FlowPilot invokes its RAG pipeline. The newer generic FlowPilotX GUI still selects from supplied candidate pools; automatically connecting original RAG/candidate generation to that GUI remains future integration work.

Fresh upstream/council generation, live literature mining and provider-backed PDF extraction need separately agreed routing, permitted inputs and budget. Saved approvals do not authorize new calls.

Detailed local logs, screenshots and reports are under the ignored `review/runtime_verification_2026-10-16_v1/` directory. Temporary verification servers and copied test corpora were stopped/removed; reusable local dependencies and built frontend assets remain installed.
