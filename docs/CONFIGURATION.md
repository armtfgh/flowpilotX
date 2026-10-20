# Configuration and local operation

## FlowPilotX without models or private data

```bash
python3 scripts/run_flowpilotx_platform.py --profile demo --port 8788
```

Python 3.11+ on Linux/macOS or WSL is sufficient. The GUI binds to `127.0.0.1`; the core uses only the standard library. `--root` identifies the study workspace, and `--storage` overrides campaign storage. Synthetic state defaults to `.flowpilotx/`.

`--profile auto` uses the synthetic profile when no private-study markers exist. A partial KHU bundle is an error; it is not silently treated as a demo. `--profile khu` requires the restored adapter and complete local private inputs. Generic and study state have separate default pointers.

## Optional FlowPilotX Qwen planner

The connector expects an owner-configured OpenAI-compatible server with `/v1/chat/completions` and `/tokenize`, serving the model name `/models/Qwen3.6-27B`. Before enabling it, establish the exact endpoint, permitted evidence and attempted-generation allowance.

```bash
export FLOWPILOTX_QWEN_BASE_URL=http://127.0.0.1:8000/v1
export FLOWPILOTX_QWEN_MAX_CALLS=1
export FLOWPILOTX_ENABLE_QWEN=1
python3 scripts/run_flowpilotx_platform.py --profile demo
```

This is an example, not authorization. The demo defaults to zero calls and a disabled route. A generation occurs only after explicit route selection and submission. The cumulative ledger counts generation attempts, including failed ones; do not reset storage to evade an allowance. There is no automatic retry or model fallback. Endpoint URLs containing embedded credentials, queries or fragments are rejected. The connector does not accept paid routes or meter self-hosted compute cost.

The generic server does not load dotenv files automatically. Archived study ledgers retain their original limits; changing the demo environment does not replenish them. Imported Astra/Sol reviews are completed records, not live generation permission.

## Original FlowPilot runtime

Create and activate a virtual environment at the repository root, then install `requirements-runtime.txt`; add `requirements-optional.txt` for literature mining. This includes the original backend requirements plus Streamlit, Plotly and pytest. The critical backend/Chroma versions exercised locally are constrained by `requirements-runtime-constraints.txt`. Remaining dependency ranges are not a fully pinned reproducibility lock; the successful local environment inherited existing packages. Record the resolved environment for a scientific run. Optional study PDF exports use `requirements-study.txt`.

Run original applications from `baseline/flowpilot/`, because several legacy paths are relative to that directory:

```bash
cd baseline/flowpilot
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

For the React client, use Node 20.19+ on the 20.x line or 22.12+, build once, then serve with the original FastAPI backend:

```bash
cd baseline/flowpilot
npm --prefix flowpilot_webapp/frontend ci
npm --prefix flowpilot_webapp/frontend run build
python -m uvicorn flowpilot_webapp.backend.app:app --host 127.0.0.1 --port 8510
```

Each block starts from the repository root. Open port 8510 with `?demo=1` for the original frontend demonstration. Install the system Graphviz `dot` executable for topology rendering. The preserved `run_dev.sh` additionally installs dependencies and binds to all interfaces; the explicit loopback commands above are preferable for local review.

## Original model routing and retrieval

The original agent retains its model catalog, configuration and provider clients. Provider credentials such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` belong in the local environment. The FastAPI application optionally loads `baseline/flowpilot/.env`; never commit it.

| Variable | Purpose | Default |
|---|---|---|
| `FLOWPILOT_OLLAMA_BASE_URL` | Original Ollama-compatible endpoint | `http://127.0.0.1:11434/v1` |
| `FLOWPILOT_QWEN36_BASE_URL` | Original Qwen 3.6 route | `http://127.0.0.1:8000/v1` |
| `FLOWPILOT_QWEN38_BASE_URL` | Original Qwen 3.8 route | `http://127.0.0.1:8001/v1` |

These original-agent variables are separate from `FLOWPILOTX_*`. A loopback address does not guarantee a model is installed. Original route availability can probe configured endpoints. Model names, providers, council execution and retrieval mode must be checked before a live design: the preserved agent has provider-specific defaults, retry behavior and availability-based default selection, and it is not governed by the generic gateway's attempt ledger.

Private runtime data paths, relative to `baseline/flowpilot/`, are:

```text
flora_translate/data/records/        Authorized source records
flora_translate/data/chroma_db/      Compatible populated Chroma index
flora_translate/data/                Local inventory profiles
flora_fundamentals/data/             Authorized handbook-derived rules
```

These paths are excluded from Git. Restore a compatible owner-held index or deliberately prepare one from authorized records. `flora_translate.main.index_records` and `VectorStore.index_folder` can generate summaries and paid embeddings; do not run them as an automatic installation or repair step. Semantic query retrieval can also call providers. The original `PipelineRuntimeOptions(retrieval_mode="lexical")` disables remote query embeddings, but the remainder of translation can still use models.

No campaign budget is granted by configuration examples, saved approvals, code restoration or credentials being present. Establish routing, permitted inputs, retry policy and spending limits for every new live campaign. The current repository does not provide a universal spending gate around every preserved legacy entry point.
