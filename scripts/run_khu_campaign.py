"""Run only an explicitly authorized, frozen KHU campaign configuration."""
from pathlib import Path
import argparse
import json

from khu_sulfoxidation.campaign import run_campaign
from khu_sulfoxidation.campaign_evidence import build_campaign_evidence
from khu_sulfoxidation.conditional_planning import build_conditional_pool
from khu_sulfoxidation.model_gateway import CampaignGateway, validate_campaign
from khu_sulfoxidation.provenance import file_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--execute", action="store_true", help="Dispatch the explicitly authorized campaign; otherwise offline preflight only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(args.config).read_text())
    validate_campaign(config)
    for name, expected in config["source_hashes"].items():
        if file_hash(root/name) != expected:
            raise ValueError(f"Authorized campaign source changed: {name}")
    pool, evidence = build_conditional_pool(root), build_campaign_evidence(root)
    directory = Path(args.campaign_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        print(json.dumps({"status":"offline_preflight_passed_no_model_calls", "pool":pool["summary"],
                          "routing":config["routing"], "maximum_cost_USD":config["maximum_cost_USD"]}, indent=2))
        return
    later = directory/"later_phase_evidence_not_sent_to_models.json"
    if not later.exists(): write_json(later,evidence["later_phase_evidence"])
    result = run_campaign(directory, config=config, pool=pool, evidence=evidence,
                          evidence_ids=evidence["visible_citation_ids"]["history_informed"], gateway_class=CampaignGateway)
    print(json.dumps({"status":"model_nominations_frozen_not_laboratory_released", "api_audit":result["api_audit"]}, indent=2), flush=True)


if __name__ == "__main__": main()
