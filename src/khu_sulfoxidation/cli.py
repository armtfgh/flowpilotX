"""Local CLI. Never imports the legacy agent runtime or contacts a provider."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .export import export_package, verify_package, zip_package
from .history import ingest_history
from .inventory import reconcile_inventory
from .provenance import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    sub = parser.add_subparsers(dest="command",required=True)
    for command in ("ingest","inventory"):
        p = sub.add_parser(command); p.add_argument("--output",type=Path,required=True)
    p = sub.add_parser("package")
    p.add_argument("--config",type=Path,default=Path("configs/study_v1.json"))
    p.add_argument("--apparatus",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pdf",action="store_true")
    p.add_argument("--zip",action="store_true")
    p.add_argument("--verification",type=Path)
    p = sub.add_parser("verify"); p.add_argument("output",type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command in ("ingest","inventory"):
        result = ingest_history(root) if args.command == "ingest" else reconcile_inventory(root)
        write_json(args.output,result)
        print(json.dumps({"written":str(args.output),"schema_version":result["schema_version"]}))
    elif args.command == "verify": print(json.dumps(verify_package(args.output),indent=2))
    else:
        config_path = args.config if args.config.is_absolute() else root/args.config
        config = json.loads(config_path.read_text())
        profile = json.loads(args.apparatus.read_text()) if args.apparatus else None
        verification = json.loads(args.verification.read_text()) if args.verification else None
        c = export_package(root,config,args.output,profile,pdf=args.pdf,verification=verification)
        result = {"output":str(args.output),"status":c["status"],"core_slots":len(c["condition_rows"]),
                  "optional_slots":len(c["optional_slots"]),"release_blocker_count":len(c["release_blockers"]),
                  "verification":verify_package(args.output)}
        if args.zip: result["archive"] = str(zip_package(args.output))
        print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
