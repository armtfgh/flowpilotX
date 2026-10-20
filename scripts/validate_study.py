#!/usr/bin/env python3
"""Run focused offline checks and preserve their commands/results in a new file."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from khu_sulfoxidation.provenance import source_snapshot, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline-python",default="python3")
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    baseline_tests=["flora_translate/tests/test_gas_settings.py","flora_translate/tests/test_inventory_constraints.py",
                    "flora_translate/tests/test_executable_artifact_contract.py","flora_translate/tests/test_scientific_gas.py"]
    isolated_runner=("import socket\ndef deny(*a,**k): raise RuntimeError('Network disabled for offline regression')\n"
                     "socket.socket.connect=deny\nsocket.create_connection=deny\nsocket.getaddrinfo=deny\n"
                     "import pytest\nraise SystemExit(pytest.main("+repr(["-q","-p","no:cacheprovider"]+baseline_tests)+"))")
    tasks=[("study_unittests",[sys.executable,"-m","unittest","discover","-s","tests","-v"],ROOT,"src"),
           ("focused_baseline_regression",[args.baseline_python,"-c",isolated_runner],ROOT/"baseline/flowpilot","."),
           ("handoff_integrity",[sys.executable,"scripts/verify_handoff.py"],ROOT,"src")]
    results=[]
    for name,command,cwd,pythonpath in tasks:
        env={**os.environ,"PYTHONPATH":pythonpath,"PYTHONDONTWRITEBYTECODE":"1",
             "PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","ANONYMIZED_TELEMETRY":"False"}
        if name=="focused_baseline_regression":
            with tempfile.TemporaryDirectory(prefix="baseline-regression-",dir=ROOT/"outputs") as temporary:
                isolated=Path(temporary)/"flowpilot"
                shutil.copytree(cwd,isolated)
                process=subprocess.run(command,cwd=isolated,env=env,text=True,capture_output=True)
        else:
            process=subprocess.run(command,cwd=cwd,env=env,text=True,capture_output=True)
        results.append({"name":name,"command":command,"cwd":str(cwd.relative_to(ROOT)),"returncode":process.returncode,
                        "stdout":process.stdout,"stderr":process.stderr,"isolated_copy":name=="focused_baseline_regression"})
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"passed" if all(r["returncode"]==0 for r in results) else "failed",
            "source_snapshot":source_snapshot(ROOT),"checks":results,
            "scope":"Deterministic offline checks only. No experiment, live model call, full baseline suite or browser build.",
            "development_attempt_notes":["An initial baseline run from workspace root failed seven relative-fixture lookups; running from baseline/flowpilot resolved them without changing baseline source.",
                "First package build preserved in draft01 failed on structured rejection deduplication; fixed and covered by integration tests.",
                "Visual QA of draft02 found zero-height diagram layout overlap; fixed before final delivery.",
                "Legacy imports added two acquire_write records in the copied Chroma DB. Original bytes restored against manifest SHA-256; no embedding-table change. Diagnostics retained in provenance/baseline_restoration_2026-10-13.json. All subsequent baseline checks use a temporary full copy with network disabled."]}
    write_json(args.output,report)
    print(json.dumps({"status":report["status"],"report":str(args.output),"checks":[{"name":r["name"],"returncode":r["returncode"]} for r in results]},indent=2))
    return 0 if report["status"]=="passed" else 1


if __name__ == "__main__": raise SystemExit(main())
