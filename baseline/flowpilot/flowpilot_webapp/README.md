# FlowPilot Workspace

This is the HILO-inspired web and desktop interface for the existing validated
FlowPilot pipeline. The browser does not duplicate chemistry or engineering
logic. It calls the same Python intake, inventory, translation, topology,
validation, autosave, and refinement modules used by the original application.

## Run locally

```bash
chmod +x flowpilot_webapp/run_dev.sh
./flowpilot_webapp/run_dev.sh
```

Open `http://localhost:8510`. A UI-only populated demonstration is available at
`http://localhost:8510/?demo=1`; it does not call an LLM.

## Architecture

- `frontend/`: React, TypeScript, Vite production client.
- `backend/app.py`: FastAPI routes for intake, inventory, asynchronous designs,
  result artifacts, refinement, and saved runs.
- `backend/jobs.py`: serialized background designs, progress messages, and
  immutable result persistence under `outputs/webapp_jobs/`.
- `backend/desktop.py`: localhost server plus native `pywebview` shell.
- `FlowPilot.spec`: Windows PyInstaller bundle.

All result views read `result.final_design` as the canonical numerical source.
The process image is served from the existing pipeline artifact. A non-executable
candidate is labelled diagnostic and never presented as a runnable design.

## Build the Windows application

PyInstaller must run on Windows to create a Windows executable. On a Windows
machine with Python 3.11 and Node 20+ installed for the build only:

```powershell
powershell -ExecutionPolicy Bypass -File flowpilot_webapp\build_windows.ps1
```

The distributable is `flowpilot_webapp/release/FlowPilot-Windows-x64.zip`.
After extraction, the end user launches `FlowPilot.exe`; Python and Node are not
required. The default selected models still require provider credentials and
network access. Put the normal `.env` beside `FlowPilot.exe` or define the keys
in the Windows environment.

The builder also embeds the Graphviz runtime used by FlowPilot's established
equipment-icon topology renderer. Graphviz is installed only on the build
runner and is not an end-user prerequisite.

The bundle includes the local corpus and Chroma index present at build time.
Build from a complete private working tree, since generated corpus files are not
stored in the public Git repository.
