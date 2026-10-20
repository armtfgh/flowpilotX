"""Bounded OpenAI Responses gateway for the explicitly authorized KHU campaign.

Only this module performs model HTTP requests. It has no fallback, retries,
retrieval, tools, or shared conversation state. Authorization must exist outside
this code; setting a configuration flag is a record, not permission itself.

Official contracts checked 2026-10-13:
https://developers.openai.com/api/docs/models/gpt-6-astra
https://developers.openai.com/api/docs/models/gpt-6.1-sol
https://developers.openai.com/api/reference/resources/responses/methods/create
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import tempfile
import urllib.error
import urllib.request
from uuid import uuid4

ENDPOINT = "https://api.openai.com/v1/responses"
TOKEN_COUNT_ENDPOINT = ENDPOINT + "/input_tokens"
ARMS = ("history_informed", "history_omitted")
COUNCIL_ROLES = ("Chemistry", "KineticsSelectivity", "FluidicsPhotoreactor", "Safety", "Skeptic", "Chief")
ROUTING = {"Upstream": "gpt-6-astra", **{role: "gpt-6.1-sol" for role in COUNCIL_ROLES}}
# Standard, short-context USD per million tokens; reserve the cache-write maximum.
PRICES = {
    "gpt-6-astra": {"input": 10, "cached_input": 1, "cache_write": 12.5, "output": 50},
    "gpt-6.1-sol": {"input": 2, "cached_input": .1, "cache_write": 2.5, "output": 10},
}


class GatewayError(RuntimeError):
    """Local preflight/integrity failure: no new request was dispatched."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _bytes_hash(value):
    return hashlib.sha256(value).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, data):
    """Exclusive, fsynced publication; never replace an audit artifact."""
    path = Path(path)
    encoded = data if isinstance(data, bytes) else canonical(data).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return _bytes_hash(encoded)


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise GatewayError(f"{label} must be an integer >= {minimum}")
    return value


def _source_hashes(value):
    if not isinstance(value, dict) or not value:
        raise GatewayError("Nonempty source_hashes must identify the exact visible/configuration sources")
    for identity, value_hash in value.items():
        if not isinstance(identity, str) or not identity or not isinstance(value_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", value_hash):
            raise GatewayError("Invalid source identity or SHA-256")


def campaign_template(*, campaign_id, authorization_reference, authorized_at, source_hashes):
    """Return the explicit fixed route/budget template; enabled authorization is opt-in."""
    return {
        "schema_version": "khu_model_gateway_v1", "campaign_id": campaign_id,
        "study_id": "khu_sulfoxidation_v1", "authorized": False,
        "authorization_reference": authorization_reference, "authorized_at": authorized_at,
        "endpoint": ENDPOINT, "service_tier": "default", "reasoning_effort": "high",
        "max_attempts": 14, "max_attempts_per_arm": 7,
        "max_input_tokens": 64000, "max_output_tokens": 16000, "maximum_cost_USD": 25,
        "automatic_retries": 0, "fallback_models": [], "tools_enabled": False,
        "automatic_reembedding": False, "routing": dict(ROUTING),
        "source_hashes": dict(source_hashes), "pricing_USD_per_million_tokens": {model:dict(prices) for model,prices in PRICES.items()},
    }


def validate_campaign(config):
    required = set(campaign_template(campaign_id="", authorization_reference="", authorized_at="", source_hashes={}))
    if not isinstance(config, dict) or set(config) != required:
        raise GatewayError("Campaign configuration must match the explicit gateway schema")
    if config["authorized"] is not True or config["study_id"] != "khu_sulfoxidation_v1":
        raise GatewayError("This KHU campaign needs recorded explicit user authorization")
    for field in ("campaign_id", "authorization_reference", "authorized_at"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise GatewayError(f"Missing {field}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config["campaign_id"]):
        raise GatewayError("campaign_id must be a plain stable identifier")
    expected = campaign_template(campaign_id="", authorization_reference="", authorized_at="", source_hashes={})
    for field in ("schema_version", "endpoint", "service_tier", "reasoning_effort", "routing", "pricing_USD_per_million_tokens", "automatic_retries", "fallback_models", "tools_enabled", "automatic_reembedding"):
        if canonical(config[field]) != canonical(expected[field]):
            raise GatewayError(f"Unauthorized gateway setting: {field}")
    for field, maximum in (("max_attempts", 14), ("max_attempts_per_arm", 7), ("max_input_tokens", 64000), ("max_output_tokens", 16000)):
        if _integer(config[field], field, 1) > maximum:
            raise GatewayError(f"{field} exceeds the approved ceiling")
    cap = config["maximum_cost_USD"]
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or not 0 < cap <= 25:
        raise GatewayError("maximum_cost_USD must be positive and <= 25")
    _source_hashes(config["source_hashes"])
    canonical(config)
    return config


def build_payload(request, model, max_output_tokens=16000):
    if not isinstance(request, dict) or set(request) - {"system_prompt", "user_prompt", "source_hashes", "response_schema"} or not {"system_prompt", "user_prompt", "source_hashes"} <= set(request):
        raise GatewayError("Request must contain exact system/user prompts and source hashes only, with optional response_schema")
    for field in ("system_prompt", "user_prompt"):
        if not isinstance(request[field], str) or not request[field].strip():
            raise GatewayError(f"{field} must be nonempty text")
    _source_hashes(request["source_hashes"])
    if model not in PRICES:
        raise GatewayError("Model is outside the authorized route")
    payload = {
        "model": model, "instructions": request["system_prompt"],
        "input": [{"role": "user", "content": [{"type": "input_text", "text": request["user_prompt"]}]}],
        "reasoning": {"effort": "high"}, "max_output_tokens": max_output_tokens,
        "service_tier": "default", "store": False, "stream": False,
        "truncation": "disabled",
    }
    if "response_schema" in request:
        if not isinstance(request["response_schema"], dict):
            raise GatewayError("response_schema must be a JSON Schema object")
        payload["text"] = {"format": {"type": "json_schema", "name": "khu_study_response", "strict": True, "schema": request["response_schema"]}}
    return payload


def conservative_input_bound(payload):
    """UTF-8 byte bound, plus 1,024 framing tokens; no approximate tokenizer claim.

    For this restricted two-message text-only request, every ordinary token has
    at least one underlying byte. Serializing the entire payload also counts
    escaped text/schema/control fields that need not all become model input.
    This intentionally rejects some longer prompts that would tokenize below
    64k. It does not silently truncate or make an extra token-count API call.
    """
    return len(canonical(payload).encode("utf-8")) + 1024


def reservation_USD(model, max_input_tokens=64000, max_output_tokens=16000):
    prices = PRICES[model]
    return float((Decimal(max_input_tokens) * Decimal(str(prices["cache_write"])) + Decimal(max_output_tokens) * Decimal(str(prices["output"]))) / Decimal(1000000))


def usage_cost(response, model, max_input_tokens, max_output_tokens):
    """Exact token-price calculation only when cache/write usage is complete."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        inputs = _integer(usage["input_tokens"], "input_tokens")
        outputs = _integer(usage["output_tokens"], "output_tokens")
        details = usage["input_tokens_details"]
        cached = _integer(details["cached_tokens"], "cached_tokens")
        writes = _integer(details["cache_write_tokens"], "cache_write_tokens")
    except (KeyError, TypeError, GatewayError):
        return None
    if cached + writes > inputs or inputs > max_input_tokens or outputs > max_output_tokens:
        return None
    prices = PRICES[model]
    total = (Decimal(inputs-cached-writes) * Decimal(str(prices["input"]))
             + Decimal(cached) * Decimal(str(prices["cached_input"]))
             + Decimal(writes) * Decimal(str(prices["cache_write"]))
             + Decimal(outputs) * Decimal(str(prices["output"]))) / Decimal(1000000)
    return float(total)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect prohibited", headers, fp)


def _http_post(payload_bytes, api_key, timeout_seconds, *, endpoint=ENDPOINT):
    if endpoint not in (ENDPOINT, TOKEN_COUNT_ENDPOINT):
        raise GatewayError("HTTP endpoint is outside the explicit allowlist")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs()
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise GatewayError("Certificate and hostname verification are required")
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), _NoRedirect())
    request = urllib.request.Request(endpoint, data=payload_bytes, method="POST", headers={
        "Authorization": "Bearer " + api_key, "Content-Type": "application/json", "Accept": "application/json"})
    with opener.open(request, timeout=timeout_seconds) as response:
        if response.geturl() != endpoint:
            raise GatewayError("Response URL differs from the exact authorized endpoint")
        return response.status, dict(response.headers), response.read()


class CampaignGateway:
    """One directory and exclusive lock account for both arms and all 14 attempts.

    A crash leaves its reserved/in-flight attempt counted at full cost and
    blocks further dispatch. No stale-lock removal or automatic recovery occurs.
    Use the same campaign archive on restart; never create a replacement archive
    to bypass an existing campaign's ledger.
    """
    def __init__(self, archive_dir, campaign_config, *, timeout_seconds=1800):
        self.config = json.loads(canonical(validate_campaign(campaign_config)))
        credential = os.environ.get("OPENAI_API_KEY", "")
        if credential and credential in canonical(self.config):
            raise GatewayError("Credential detected in configuration; nothing was archived")
        self.config_hash = digest(self.config)
        self.root = Path(archive_dir).resolve()
        self.timeout = timeout_seconds
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise GatewayError("timeout_seconds must be finite and positive")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock():
            path = self.root / "configuration.json"
            if not path.exists():
                if set(p.name for p in self.root.iterdir()) != {".gateway.lock"}:
                    raise GatewayError("Archive has data but no configuration; refusing a reset")
                _write(path, self.config)
                (self.root / "ledger").mkdir()
                (self.root / "attempts").mkdir()
                (self.root / "preflight_rejections").mkdir()
                (self.root / "token_counts").mkdir()
                self._save_ledger({"configuration_sha256": self.config_hash, "entries": []}, None)
            elif path.read_bytes() != canonical(self.config).encode():
                raise GatewayError("Existing campaign configuration differs; no routing/budget changes mid-campaign")
            self._load_ledger()

    @contextmanager
    def _lock(self):
        lock = self.root / ".gateway.lock"
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise GatewayError("Campaign is locked; concurrent execution or unresolved interrupted process") from exc
        try:
            yield
        finally:
            lock.rmdir()

    def _save_ledger(self, state, previous):
        sequence = 0 if previous is None else previous["sequence"] + 1
        body = {"sequence": sequence, "previous_sha256": None if previous is None else digest(previous), "saved_at": _now(), **state}
        body = json.loads(canonical(body))
        _write(self.root / "ledger" / f"{sequence:06d}.json", body)
        return body

    def _load_ledger(self):
        snapshots = sorted((self.root / "ledger").iterdir())
        previous = None
        if not snapshots:
            raise GatewayError("Missing campaign ledger; refusing a reset")
        for sequence, path in enumerate(snapshots):
            try:
                current = json.loads(path.read_text())
                if path.name != f"{sequence:06d}.json" or current["sequence"] != sequence or current["previous_sha256"] != (None if previous is None else digest(previous)) or current["configuration_sha256"] != self.config_hash:
                    raise GatewayError("Ledger sequence/hash/configuration mismatch")
            except (ValueError, KeyError, TypeError) as exc:
                raise GatewayError("Corrupt campaign ledger; refusing further calls") from exc
            previous = current
        directories = set(p.name for p in (self.root / "attempts").iterdir())
        if directories != {entry["attempt_id"] for entry in previous["entries"]}:
            raise GatewayError("Attempt/ledger mismatch; interrupted write or deleted ledger")
        for entry in previous["entries"]:
            folder = self.root / "attempts" / entry["attempt_id"]
            actual = set(p.name for p in folder.iterdir())
            if actual != set(entry["files_sha256"]):
                raise GatewayError("Attempt files differ from its ledger inventory")
            for name, expected_hash in entry["files_sha256"].items():
                if _bytes_hash((folder / name).read_bytes()) != expected_hash:
                    raise GatewayError("Attempt artifact hash mismatch")
        return previous

    def _count_input(self, payload, api_key):
        """One official count request; no fallback or generation on count failure.

        The count endpoint accepts instructions, input and text.format, so the
        exact structured-output schema is counted too. Add 1,024 tokens of
        conservative framing headroom. Archives distinguish this non-generation
        operation from the 14 paid generation attempts.
        """
        folder = self.root / "token_counts" / uuid4().hex
        folder.mkdir()
        request = {key: payload[key] for key in ("model", "instructions", "input", "text") if key in payload}
        files = {"request.json": _write(folder / "request.json", request)}
        files["dispatch.json"] = _write(folder / "dispatch.json", {"at":_now(), "endpoint":TOKEN_COUNT_ENDPOINT, "generation":False})
        status, raw, error = None, b"", None
        try:
            status, _, raw = _http_post(canonical(request).encode(), api_key, self.timeout, endpoint=TOKEN_COUNT_ENDPOINT)
        except urllib.error.HTTPError as exc:
            status = exc.code
            try: raw = exc.read()
            except Exception: raw = b""
            error = "HTTP token-count request failed"
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc).replace(api_key, "[credential redacted]")
        received_hash = _bytes_hash(raw)
        redacted = api_key.encode() in raw
        raw = raw.replace(api_key.encode(), b"[credential redacted]")
        files["response.raw"] = _write(folder / "response.raw", raw)
        count = None
        try:
            if error or status != 200 or redacted:
                raise GatewayError(error or "Invalid token-count HTTP response")
            parsed = json.loads(raw)
            if parsed.get("object") != "response.input_tokens":
                raise GatewayError("Token-count response has the wrong object type")
            count = _integer(parsed["input_tokens"], "input token count")
        except (ValueError, KeyError, TypeError, GatewayError) as exc:
            error = str(exc).replace(api_key,"[credential redacted]")
        manifest = {"completed_at":_now(), "endpoint":TOKEN_COUNT_ENDPOINT,
                    "files_sha256":files, "received_sha256":received_hash,
                    "input_tokens":count, "framing_headroom_tokens":1024,
                    "error":error, "generation":False, "credential_redacted":redacted}
        _write(folder / "manifest.json", manifest)
        if error:
            raise GatewayError("Official input token count failed; generation was not dispatched")
        return count + 1024, {"basis":"official_count_including_text_schema_plus_1024_headroom", "input_tokens":count, "archive_path":str(folder)}

    def _advance(self, ledger, entry):
        entries = ledger["entries"]
        if entries and entries[-1]["attempt_id"] == entry["attempt_id"]:
            entries = entries[:-1] + [entry]
        else:
            entries = entries + [entry]
        return self._save_ledger({"configuration_sha256": self.config_hash, "entries": entries}, ledger)

    @staticmethod
    def _accounted(ledger):
        return sum((Decimal(str(entry["accounted_cost_USD"])) for entry in ledger["entries"]), Decimal(0))

    def audit(self):
        with self._lock():
            ledger = self._load_ledger()
            entries = ledger["entries"]
            known = sum((Decimal(str(e["cost_USD"])) for e in entries if e.get("cost_USD") is not None), Decimal(0))
            accounted = self._accounted(ledger)
            counts = list((self.root / "token_counts").iterdir())
            for folder in counts:
                try:
                    manifest = json.loads((folder / "manifest.json").read_text())
                    if set(p.name for p in folder.iterdir()) != set(manifest["files_sha256"]) | {"manifest.json"}:
                        raise GatewayError("Token-count archive inventory mismatch")
                    if any(_bytes_hash((folder / name).read_bytes()) != expected for name,expected in manifest["files_sha256"].items()):
                        raise GatewayError("Token-count artifact hash mismatch")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise GatewayError("Incomplete/corrupt token-count archive") from exc
            return {"campaign_id": self.config["campaign_id"], "configuration_sha256": self.config_hash,
                    "input_count_operations":len(counts),
                    "attempt_count": len(entries), "per_arm_attempts": {arm: sum(e["arm"] == arm for e in entries) for arm in ARMS},
                    "known_token_cost_USD": float(known), "accounted_cost_USD": float(accounted),
                    "unknown_or_reserved_attempts": [e["attempt_id"] for e in entries if e.get("cost_USD") is None or e["status"] != "completed"],
                    "remaining_budget_USD": float(Decimal(str(self.config["maximum_cost_USD"]))-accounted),
                    "unresolved_inflight": [e["attempt_id"] for e in entries if e["status"] in ("reserved", "in_flight")],
                    "entries": entries, "ledger_sha256": digest(ledger), "cost_basis": "Token pricing before taxes; incomplete/failed attempts retain their full reservation."}

    def execute_request(self, request, *, role, arm):
        """Return a result with ok=False for retained failed API attempts; no retry.

        Local preflight refusals raise GatewayError, are archived separately and
        do not consume a paid attempt. Check result['ok'] before downstream use.
        """
        with self._lock():
            ledger = self._load_ledger()
            api_key = os.environ.get("OPENAI_API_KEY", "")
            try:
                if role not in ROUTING or arm not in ARMS:
                    raise GatewayError("Unknown campaign role or comparison arm")
                if not api_key or not api_key.strip() or "\n" in api_key or "\r" in api_key:
                    raise GatewayError("OPENAI_API_KEY is not available as a valid environment secret")
                if api_key in canonical(request):
                    raise GatewayError("Credential detected in request: raw payload intentionally not archived")
                payload = build_payload(request, self.config["routing"][role], self.config["max_output_tokens"])
                bound = conservative_input_bound(payload)
                input_guard = {"basis":"utf8_bytes_of_entire_payload_plus_1024_headroom", "upper_bound":bound}
                entries = ledger["entries"]
                if any(e["status"] in ("reserved", "in_flight") or e.get("halt_campaign") for e in entries):
                    raise GatewayError("An unresolved attempt or billing/route mismatch blocks new dispatch")
                if len(entries) >= self.config["max_attempts"] or sum(e["arm"] == arm for e in entries) >= self.config["max_attempts_per_arm"]:
                    raise GatewayError("Campaign or per-arm attempt budget exhausted")
                if any(e["role"] == role and e["arm"] == arm for e in entries):
                    raise GatewayError("Role/arm already attempted; no retry or revision allowance remains")
                reservation = reservation_USD(payload["model"], self.config["max_input_tokens"], self.config["max_output_tokens"])
                if self._accounted(ledger) + Decimal(str(reservation)) > Decimal(str(self.config["maximum_cost_USD"])):
                    raise GatewayError("Cannot reserve the full worst-case request cost within the campaign cap")
                if bound > self.config["max_input_tokens"]:
                    bound, input_guard = self._count_input(payload, api_key)
                if bound > self.config["max_input_tokens"]:
                    raise GatewayError(f"Input token count plus framing headroom {bound} exceeds {self.config['max_input_tokens']}; no generation or truncation")
            except (ValueError, TypeError, GatewayError) as exc:
                error = str(exc).replace(api_key, "[credential redacted]") if api_key else str(exc)
                rejection = {"at": _now(), "role": role, "arm": arm, "error": error, "dispatched": False, "configuration_sha256": self.config_hash}
                # Retain exact rejected proposals except when they contain a credential.
                if isinstance(request, dict) and (not api_key or api_key not in canonical(request)):
                    rejection["request"] = request
                _write(self.root / "preflight_rejections" / f"{uuid4().hex}.json", rejection)
                raise GatewayError(error) from None
            attempt_id = f"attempt-{len(ledger['entries'])+1:04d}"
            folder = self.root / "attempts" / attempt_id
            folder.mkdir()
            entry = {"attempt_id": attempt_id, "role": role, "arm": arm, "model": payload["model"], "status": "reserved",
                     "created_at": _now(), "reserved_cost_USD": reservation, "accounted_cost_USD": reservation, "cost_USD": None,
                     "input_token_upper_bound": bound, "input_guard":input_guard, "files_sha256": {}}
            for name, value in {"input.json": request, "request.json": payload, "reservation.json": {k:v for k,v in entry.items() if k != "files_sha256"}}.items():
                entry["files_sha256"][name] = _write(folder / name, value)
            ledger = self._advance(ledger, entry)
            entry["files_sha256"]["dispatch.json"] = _write(folder / "dispatch.json", {"at": _now(), "endpoint": ENDPOINT, "headers": {"Content-Type": "application/json", "Accept": "application/json"}, "authorization_header_archived": False})
            entry["status"] = "in_flight"
            ledger = self._advance(ledger, entry)
            status_code, headers, raw, caught = None, {}, b"", None
            try:
                status_code, headers, raw = _http_post(canonical(payload).encode("utf-8"), api_key, self.timeout)
            except urllib.error.HTTPError as exc:
                status_code, headers = exc.code, dict(exc.headers or {})
                raw = exc.read()
                caught = {"type": type(exc).__name__, "message": "HTTP request failed", "status_code": status_code}
            except BaseException as exc:
                caught = {"type": type(exc).__name__, "message": str(exc).replace(api_key, "[credential redacted]")}
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    caught["interrupted"] = True
            received_hash = _bytes_hash(raw)
            redacted = api_key.encode() in raw
            safe_raw = raw.replace(api_key.encode(), b"[credential redacted]")
            entry["files_sha256"]["response.raw"] = _write(folder / "response.raw", safe_raw)
            response, text, parsed, usage, cost = None, "", None, None, None
            errors = [caught] if caught else []
            halt = redacted
            try:
                if caught or status_code != 200:
                    raise GatewayError("Request failed after dispatch; full reservation retained")
                response = json.loads(safe_raw)
                if not isinstance(response, dict):
                    raise GatewayError("Response body is not an object")
                usage = response.get("usage")
                if response.get("model") != payload["model"] or response.get("service_tier") != "default":
                    halt = True
                    raise GatewayError("Returned model/service tier differs from the authorized route")
                if isinstance(usage, dict) and (type(usage.get("input_tokens")) is int and usage["input_tokens"] > self.config["max_input_tokens"] or type(usage.get("output_tokens")) is int and usage["output_tokens"] > self.config["max_output_tokens"]):
                    halt = True
                    raise GatewayError("Provider usage exceeds approved per-call limits")
                cost = usage_cost(response, payload["model"], self.config["max_input_tokens"], self.config["max_output_tokens"])
                if response.get("status") != "completed":
                    raise GatewayError("Response is incomplete, failed or cancelled; no continuation/retry")
                text = "".join(part["text"] for output in response.get("output", []) if output.get("type") == "message" for part in output.get("content", []) if part.get("type") == "output_text")
                if not text:
                    raise GatewayError("No completed output text")
                parsed = json.loads(text)
                if redacted:
                    raise GatewayError("Credential appeared in response; redacted and halted")
            except (ValueError, KeyError, TypeError, GatewayError) as exc:
                errors.append({"type": type(exc).__name__, "message": str(exc).replace(api_key, "[credential redacted]")})
            ok = not errors
            entry.update(status="completed" if ok else "failed", completed_at=_now(), cost_USD=cost,
                         accounted_cost_USD=cost if ok and cost is not None else reservation, halt_campaign=halt)
            result = {"ok": ok, "attempt_id": attempt_id, "archive_path": str(folder), "role": role, "arm": arm,
                      "model": payload["model"], "status": entry["status"], "text": text, "parsed_json": parsed,
                      "usage": usage, "cost_USD": cost, "reserved_cost_USD": reservation,
                      "accounted_cost_USD": entry["accounted_cost_USD"], "errors": errors, "halt_campaign": halt,
                      "response_status_code": status_code, "response_sha256_received": received_hash,
                      "credential_redacted": redacted, "request_id": str(headers.get("x-request-id", "")).replace(api_key,"[credential redacted]")}
            entry["files_sha256"]["result.json"] = _write(folder / "result.json", result)
            self._advance(ledger, entry)
            if caught and caught.get("interrupted"):
                raise KeyboardInterrupt("Campaign request interrupted; attempt retained at full reserved cost")
            return result
