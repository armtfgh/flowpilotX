"""Persistent, append-only scientific campaign state and frozen experiment rounds."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from .evidence import citation_ids, digest, json_copy, text, validate_evidence, validate_question, validate_result
from .inventory import validate_inventory, validate_requirements
from .selection import select, validate_candidates, validate_observation_contract

SCHEMA = "flowpilotx-campaign-v1"
SAFE_ID = re.compile(r"campaign-[a-f0-9]{24}\Z")
NON_ENGINEERING_FIELDS = {"name", "description", "protocol_text", "research_notes", "objective", "evidence", "questions",
                          "source_references", "source_hashes", "model_config", "model_evidence", "model_evidence_sha256",
                          "evidence_sha256", "candidates", "candidate_pool_context_sha256"}


class CampaignError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {"status": "rejected", "rejection_reasons": [message]}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _engineering_context(spec):
    return {k: v for k, v in spec.items() if k not in NON_ENGINEERING_FIELDS and not k.startswith("model_")}


def _contract(spec):
    return digest({"engineering": _engineering_context(spec), "candidates": spec["candidates"], "objective": spec["objective"]})


def _domain(spec):
    return digest({"chemistry": spec["chemistry"], "observation_domain": spec.get("observation_domain")})


def _merge(original, patch):
    result = deepcopy(original)
    for key, value in patch.items():
        # Compact model packets are derived snapshots, never partial dictionaries.
        result[key] = _merge(result[key], value) if not key.startswith("model_") and isinstance(result.get(key), dict) and isinstance(value, dict) else deepcopy(value)
    return result


def validate_spec(spec, previous=None):
    s = json_copy(spec)
    if not isinstance(s, dict):
        raise ValueError("Campaign specification must be an object")
    text(s.get("name"), "campaign name")
    if not isinstance(s.get("chemistry"), dict) or not s["chemistry"]:
        raise ValueError("An explicit chemistry description is required; no chemistry is inferred")
    objective = s.get("objective")
    if not isinstance(objective, dict) or objective.get("mode") not in {"optimize", "explore", "generalize"}:
        raise ValueError("objective.mode must be optimize, explore or generalize")
    if objective.get("direction", "maximize") not in {"maximize", "minimize"}:
        raise ValueError("Objective direction must be maximize or minimize")
    if objective["mode"] == "optimize":
        text(objective.get("metric"), "objective analyte metric")
        text(objective.get("unit"), "objective unit")
    s["inventory"] = validate_inventory(s.get("inventory", {"items": []}))
    s["requirements"] = validate_requirements(s.get("requirements", []))
    s["evidence"] = validate_evidence(s.get("evidence", []))
    s["evidence_sha256"] = digest(s["evidence"])
    if "model_evidence" in s:
        if s.get("model_evidence_sha256") not in {None, s["evidence_sha256"]}:
            raise ValueError("Compact model evidence is bound to a different canonical evidence version")
        s["model_evidence_sha256"] = s["evidence_sha256"]
    if not isinstance(s.get("questions", []), list):
        raise ValueError("questions must be a list")
    old = {q["id"]: q for q in previous.get("questions", [])} if previous else {}
    questions = []
    for q in s.get("questions", []):
        prior = old.get(q.get("id", q.get("question_id"))) if isinstance(q, dict) else None
        questions.append(deepcopy(q) if prior == q else validate_question(q, prior))
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    s["questions"] = questions
    s["candidates"] = validate_candidates(s.get("candidates", []), {q["id"] for q in questions})
    if "observation_contract" in s:
        s["observation_contract"] = validate_observation_contract(s["observation_contract"], s["candidates"])
    for key in ["protocol_text", "research_notes"]:
        if key in s and not isinstance(s[key], str):
            raise ValueError(key + " must be text")
    for key in ["source_references"]:
        if not isinstance(s.get(key, []), list):
            raise ValueError(key + " must be a list")
    s["candidate_pool_context_sha256"] = digest(_engineering_context(s))
    return s


def _check_amendment_references(question, spec, results):
    amendment = question.get("amendment")
    if amendment:
        known = citation_ids(spec["evidence"], spec.get("source_references", []), results)
        missing = set(amendment["evidence_ids"]) - known
        if missing:
            raise ValueError("Question amendment cites unarchived evidence: " + ", ".join(sorted(missing)))


class CampaignStore:
    """Use one storage directory for campaigns; no network/model/equipment calls.

    Revisions and rounds are new exclusive files. `get` adds computed views of
    stale contracts without rewriting old recommendations or observed results.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self):
        with (self.root / ".campaign-store.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _directory(self, ident):
        if not isinstance(ident, str) or not SAFE_ID.fullmatch(ident):
            raise CampaignError("Invalid campaign ID")
        directory = self.root / ident
        if not directory.is_dir():
            raise CampaignError("Campaign not found")
        return directory

    @staticmethod
    def _write(path, data):
        snapshot = json_copy(data)
        snapshot["content_sha256"] = digest({k: v for k, v in snapshot.items() if k != "content_sha256"})
        raw = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic publication without replacing an existing frozen snapshot.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return snapshot

    @staticmethod
    def _read(path):
        value = json.loads(path.read_text())
        if digest({k: v for k, v in value.items() if k != "content_sha256"}) != value.get("content_sha256"):
            raise CampaignError("Stored snapshot integrity check failed: " + path.name)
        return value

    def _current(self, ident):
        directory = self._directory(ident)
        revisions = sorted((directory / "revisions").glob("*.json"))
        if not revisions:
            raise CampaignError("Campaign has no revision")
        previous = None
        current = None
        for index, path in enumerate(revisions, 1):
            current = self._read(path)
            if current["version"] != index or current.get("previous_revision_sha256") != previous:
                raise CampaignError("Campaign revision chain is incomplete or inconsistent")
            previous = current["content_sha256"]
        return current

    def _rounds(self, ident):
        return sorted([self._read(p) for p in (self._directory(ident) / "rounds").glob("*.json")],
                      key=lambda r: (r["created_at"], r["id"]))

    def _append(self, prior, spec, results, event):
        data = {"schema_version": SCHEMA, "id": prior["id"], "name": spec["name"], "version": prior["version"] + 1,
                "spec": spec, "results": results, "created_at": prior["created_at"], "updated_at": _now(),
                "contract_sha256": _contract(spec), "observation_domain_sha256": _domain(spec),
                "previous_revision_sha256": prior["content_sha256"], "event": event,
                "laboratory_released": False}
        path = self._directory(prior["id"]) / "revisions" / f"{data['version']:06d}.json"
        return self._write(path, data)

    def create(self, spec):
        s = validate_spec(spec)
        with self._lock():
            ident = "campaign-" + uuid4().hex[:24]
            directory = self.root / ident
            (directory / "revisions").mkdir(parents=True, exist_ok=False)
            (directory / "rounds").mkdir()
            created = _now()
            data = {"schema_version": SCHEMA, "id": ident, "name": s["name"], "version": 1,
                    "spec": s, "results": [], "created_at": created, "updated_at": created,
                    "contract_sha256": _contract(s), "observation_domain_sha256": _domain(s),
                    "previous_revision_sha256": None, "event": {"type": "created"}, "laboratory_released": False}
            self._write(directory / "revisions" / "000001.json", data)
        return self.get(ident)

    def get(self, ident):
        current = self._current(ident)
        rounds = self._rounds(ident)
        for r in rounds:
            r["current_contract_status"] = "matches_current_contract" if r["contract_sha256"] == current["contract_sha256"] else "superseded_contract_preserved"
            frozen_spec = r["configuration"]
            changes = []
            for key in ["evidence", "questions", "protocol_text", "research_notes"]:
                if digest(frozen_spec.get(key)) != digest(current["spec"].get(key)):
                    changes.append(key)
            if digest(r.get("results_snapshot", [])) != digest(current["results"]):
                changes.append("measured_results")
            r["scientific_context_changes_since_freeze"] = changes
            r["current_scientific_context_status"] = "changed_context_preserved_prospective_snapshot" if changes else "matches_current_scientific_context"
        current.update(rounds=rounds, last_round=rounds[-1] if rounds else None,
                       revision_count=current["version"], protocol_interpretation="stored_context_only_unless_translated_into_reviewed_structured_requirements")
        return current

    def list(self):
        summaries = []
        for path in sorted(self.root.glob("campaign-*")):
            if not path.is_dir() or not SAFE_ID.fullmatch(path.name):
                continue
            c = self.get(path.name)
            summaries.append({k: c[k] for k in ["id", "name", "version", "created_at", "updated_at"]} | {
                "objective": c["spec"]["objective"], "round_count": len(c["rounds"]), "result_count": len(c["results"]),
                "last_round": c["last_round"], "laboratory_released": False})
        return summaries

    def revise(self, ident, patch):
        p = json_copy(patch)
        if not isinstance(p, dict) or not p:
            raise CampaignError("Revision needs a nonempty specification patch")
        if set(p) & {"id", "version", "results", "rounds", "content_sha256"}:
            raise CampaignError("Revision patches edit specification fields only; results use add_results")
        with self._lock():
            prior = self._current(ident)
            spec = _merge(prior["spec"], p)
            evidence_changed = digest(spec.get("evidence", [])) != digest(prior["spec"].get("evidence", []))
            invalidated_views = []
            for canonical, compact in [("evidence", "model_evidence"), ("questions", "model_questions"),
                                       ("candidates", "model_candidates"), ("inventory", "model_inventory")]:
                if digest(spec.get(canonical)) != digest(prior["spec"].get(canonical)) and compact not in p:
                    if compact in spec:
                        invalidated_views.append(compact)
                    spec.pop(compact, None)
            if evidence_changed:
                spec.pop("model_evidence_sha256", None)
                if "model_evidence" not in p:
                    spec.pop("model_evidence", None)
            changed = digest(_engineering_context(spec)) != digest(_engineering_context(prior["spec"]))
            invalidated = changed and "candidates" not in p
            if invalidated:
                spec.pop("model_candidates", None)
                for candidate in spec["candidates"]:
                    candidate["feasible"] = False
                    candidate["engineering_status"] = "unsupported"
                    candidate["rejection_reasons"] = list(dict.fromkeys(candidate["rejection_reasons"] + [
                        "Engineering context changed after candidate creation; obtain a newly checked candidate pool"]))
            spec = validate_spec(spec, prior["spec"])
            previous_questions = {q["id"]: q for q in prior["spec"]["questions"]}
            for q in spec["questions"]:
                if previous_questions.get(q["id"]) != q:
                    _check_amendment_references(q, spec, prior["results"])
            self._append(prior, spec, prior["results"], {"type": "spec_revised", "patch": p,
                "old_candidates_invalidated": invalidated, "replacement_candidate_pool_explicit": "candidates" in p,
                "compact_evidence_invalidated": evidence_changed and "model_evidence" not in p,
                "invalidated_compact_views": invalidated_views,
                "prose_enforcement": "Protocol/notes are context, not automatically parsed constraints"})
        return self.get(ident)

    def add_question(self, ident, question):
        with self._lock():
            prior = self._current(ident)
            qid = question.get("id", question.get("question_id")) if isinstance(question, dict) else None
            previous = next((q for q in prior["spec"]["questions"] if q["id"] == qid), None)
            q = validate_question(question, previous)
            _check_amendment_references(q, prior["spec"], prior["results"])
            spec = deepcopy(prior["spec"])
            spec["questions"] = [q if old["id"] == qid else old for old in spec["questions"]] if previous else spec["questions"] + [q]
            spec.pop("model_questions", None)
            self._append(prior, spec, prior["results"], {"type": "question_amended" if previous else "question_added",
                         "question_id": q["id"], "prior_question_sha256": digest(previous) if previous else None,
                         "assessment": "documented_qualitative_no_automatic_posterior_update"})
        return self.get(ident)

    def add_results(self, ident, records):
        if not isinstance(records, list) or not records:
            raise CampaignError("Results must be a nonempty list of measured records")
        with self._lock():
            prior, rounds = self._current(ident), self._rounds(ident)
            results = deepcopy(prior["results"])
            ids = {r["id"] for r in results}
            additions = []
            for record in records:
                r = validate_result(record, rounds)
                if r["id"] in ids:
                    raise CampaignError("Result ID already exists; corrections need a new ID with supersedes/reason")
                if r.get("supersedes"):
                    if r["supersedes"] not in ids:
                        raise CampaignError("Corrected result references an unknown prior result")
                    text(r.get("correction_reason"), "correction_reason")
                ids.add(r["id"])
                frozen = next(x for x in rounds if x["id"] == r["round_id"])
                r["source_observation_domain_sha256"] = frozen["observation_domain_sha256"]
                r["observation_contract"] = deepcopy(frozen["configuration"].get("observation_contract"))
                r["source_observation_contract_sha256"] = digest(r["observation_contract"])
                r["result_sha256"] = digest({k: v for k, v in r.items() if k != "result_sha256"})
                additions.append(r)
            results.extend(additions)
            self._append(prior, prior["spec"], results, {"type": "results_added", "result_ids": [r["id"] for r in additions],
                         "question_status_changes": [], "failure_policy": "No zero imputation or automatic causal attribution"})
        return self.get(ident)

    def generate(self, ident, request):
        with self._lock():
            current = self._current(ident)
            round_id = "round-" + uuid4().hex[:24]
            base = {"schema_version": "flowpilotx-round-v1", "id": round_id, "campaign_id": ident,
                    "campaign_version": current["version"], "created_at": _now(),
                    "contract_sha256": current["contract_sha256"], "observation_domain_sha256": current["observation_domain_sha256"],
                    "campaign_revision_sha256": current["content_sha256"], "configuration": deepcopy(current["spec"]),
                    "questions_snapshot": deepcopy(current["spec"]["questions"]),
                    "source_hashes": {"spec_sha256": digest(current["spec"]), "declared_sources": current["spec"].get("source_hashes", {}),
                        "implementation": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}},
                    "laboratory_released": False, "flow_outcomes_predicted": False}
            try:
                base["request"] = json_copy(request)
                base["request_sha256"] = digest(request)
                expected_version = request.get("expected_campaign_version") if isinstance(request, dict) else None
                expected_revision = request.get("expected_revision_sha256") if isinstance(request, dict) else None
                if expected_version is not None and (type(expected_version) is not int or expected_version != current["version"]):
                    raise ValueError("Campaign version changed since planning; proposal must be reviewed against the new context")
                if expected_revision is not None and expected_revision != current["content_sha256"]:
                    raise ValueError("Campaign revision hash changed since planning; refusing to bind old model reasoning to new settings")
                superseded = {r["supersedes"] for r in current["results"] if r.get("supersedes")}
                active_results = [r for r in current["results"] if r["id"] not in superseded]
                selection = select(current["spec"], active_results, request, current["observation_domain_sha256"])
                base.update(selection)
                base["results_snapshot"] = deepcopy(current["results"])
                base["superseded_result_ids_excluded"] = sorted(superseded)
                for candidate in base["selected"]:
                    candidate["question_cards"] = [q for q in base["questions_snapshot"] if q["id"] in candidate["question_ids"]]
            except (ValueError, TypeError, KeyError) as exc:
                base.update(status="rejected", selected=[], repeats=[], candidate_pool=deepcopy(current["spec"]["candidates"]),
                            rejection_reasons=[str(exc)], invalid_request_repr=repr(request) if "request" not in base else None)
            frozen = self._write(self._directory(ident) / "rounds" / (round_id + ".json"), base)
            if frozen["status"] == "rejected":
                raise CampaignError("; ".join(frozen["rejection_reasons"]), frozen)
            return frozen
