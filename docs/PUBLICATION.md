# Preparing this existing repository for GitHub

Publish the source from this existing `FlowPilotX` directory. There is no nested release repository. Cleanup separates reusable software from private data and bulky generated work; it does not remove the original agent or study adapter.

## Source and private files

| Retained for source review | Local only; excluded from Git |
|---|---|
| Original FlowPilot RAG, embeddings, vector-store, councils and engineering | Populated literature records, Chroma indexes and handbook-derived content |
| Topology renderer and equipment artwork, subject to rights review | Laboratory source records, inventories and correspondence |
| Original Streamlit and FastAPI/React interfaces | Model prompts/responses, approvals, costs and saved campaign state |
| Generic FlowPilotX engine, GUI and KHU adapter software | Generated slides, diagrams, screenshots, reports and ZIPs |
| Synthetic examples, public documentation and offline tests | Private study notes, provenance archives, environments, caches and temporary builds |

Private inputs and selected active study artifacts remain available in the local workspace. The complete original archive is also retained outside it. A public clone has the synthetic demo; actual retrieval and study reproduction require separately supplied authorized inputs. Do not upload a ZIP of the entire local working directory: that would include ignored files.

The pristine archive retains the original handoff. The documented baseline portability/runtime patches and their original hashes/diffs are recorded locally under `provenance/baseline_*patches_*/`. Recovered supplemental modules have their own source-hash manifest. The public source uses configurable loopback endpoints and generic example paths.

## Check the publication boundary

```bash
python3 tools/run_public_tests.py
python3 tools/check_release.py
git status --short
git diff --cached --stat
git diff --cached --name-only
```

Review the actual files selected for Git. `.gitignore` does not hide files already tracked or remove information from history. Historical Git objects are preserved privately; inspect the current repository history and remote before publication. A source scan is a useful check, not proof that every sensitive detail has been found.

The public tests use synthetic fixtures and mocked model transports. Full private-study tests require ignored inputs. Selected original-component checks are distinct from the entire original regression suite, which includes references to external historical fixtures not supplied here.

## License and publication

Choose a software license after confirming authorship and redistribution rights. Equipment icons and other third-party assets need a rights decision too. Keeping artwork in the complete local source package does not establish permission to publish it. See [LICENSE_STATUS.md](../LICENSE_STATUS.md).

When the file review and rights decisions are complete, the owner can commit and connect an empty GitHub repository:

```bash
git add .
git diff --cached --stat
git commit -m "Add FlowPilot and FlowPilotX source"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

These are instructions only. Restoration and preparation do not create a remote, push files, grant a license or authorize model use. Check an existing remote with `git remote -v` before adding one. Continue reviewing every future staged diff so private records and generated state remain local.
