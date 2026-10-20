"""Every transport is mocked; these tests never contact a model provider."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import ssl
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from khu_sulfoxidation.model_gateway import (
    ARMS, COUNCIL_ROLES, ENDPOINT, TOKEN_COUNT_ENDPOINT, CampaignGateway,
    GatewayError, _http_post, build_payload, campaign_template, canonical,
    conservative_input_bound, reservation_USD, usage_cost, validate_campaign,
)

FAKE_KEY = "synthetic-unit-test-credential-never-real"
HASH = "a" * 64


def config_fixture(**changes):
    config = campaign_template(campaign_id="synthetic-gateway-test", authorization_reference="Synthetic test fixture, not real spending authorization", authorized_at="2026-10-13T12:00:00+09:00", source_hashes={"fixture-inputs":HASH})
    config.update(authorized=True, **changes)
    return config


def request_fixture(**changes):
    return {"system_prompt":"Return only the requested JSON for this synthetic fixture.", "user_prompt":"Choose no real chemical conditions. Return {\"fixture\":true}.", "source_hashes":{"fixture-inputs":HASH}, **changes}


def mock_response(payload, key, timeout, **kwargs):
    model = json.loads(payload)["model"]
    if kwargs.get("endpoint") == TOKEN_COUNT_ENDPOINT:
        return 200, {}, canonical({"object":"response.input_tokens", "input_tokens":20000}).encode()
    response = {"id":"resp_fixture", "model":model, "service_tier":"default", "status":"completed",
                "output":[{"type":"message", "content":[{"type":"output_text", "text":"{\"fixture\":true}"}]}],
                "usage":{"input_tokens":1000,"input_tokens_details":{"cached_tokens":100,"cache_write_tokens":200}, "output_tokens":100, "output_tokens_details":{"reasoning_tokens":20}, "total_tokens":1100}}
    return 200, {"x-request-id":"fixture-request"}, canonical(response).encode()


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"campaign"
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY":FAKE_KEY})
        self.env.start()
        self.http = patch("khu_sulfoxidation.model_gateway._http_post", side_effect=mock_response)
        self.transport = self.http.start()

    def tearDown(self):
        self.http.stop(); self.env.stop(); self.temp.cleanup()

    def gateway(self, **changes):
        return CampaignGateway(self.root, config_fixture(**changes))

    def test_authorization_and_settings_are_explicit_and_fixed(self):
        template = campaign_template(campaign_id="test",authorization_reference="fixture",authorized_at="fixture",source_hashes={"fixture":HASH})
        self.assertFalse(template["authorized"])
        for field, value in [("authorized",False),("endpoint","https://example.com/responses"),("reasoning_effort","low"),("service_tier","auto"),("automatic_retries",1),("maximum_cost_USD",26),("max_attempts",15),("max_attempts_per_arm",8),("max_input_tokens",64001),("max_output_tokens",16001),("tools_enabled",True)]:
            config=config_fixture();config[field]=value
            with self.subTest(field=field), self.assertRaises(GatewayError): validate_campaign(config)
        config=config_fixture();config["routing"]["Upstream"]="gpt-6.1-sol"
        with self.assertRaises(GatewayError):validate_campaign(config)
        config=config_fixture();config["pricing_USD_per_million_tokens"]["gpt-6-astra"]["input"]=0
        with self.assertRaises(GatewayError):validate_campaign(config)
        self.transport.assert_not_called()

    def test_exact_safe_payload_and_reservations(self):
        payload=build_payload(request_fixture(),"gpt-6-astra")
        self.assertEqual(payload["reasoning"],{"effort":"high"})
        self.assertFalse(payload["store"])
        self.assertEqual(payload["service_tier"],"default")
        self.assertEqual(payload["max_output_tokens"],16000)
        self.assertTrue({"seed","temperature","tools","previous_response_id","conversation","prompt_cache_key"}.isdisjoint(payload))
        self.assertEqual(conservative_input_bound(payload),len(canonical(payload).encode())+1024)
        self.assertAlmostEqual(reservation_USD("gpt-6-astra"),1.6)
        self.assertAlmostEqual(reservation_USD("gpt-6.1-sol"),.32)
        self.assertAlmostEqual(2*reservation_USD("gpt-6-astra")+12*reservation_USD("gpt-6.1-sol"),7.04)

    def test_success_preserves_exact_payload_cost_and_restart(self):
        gateway=self.gateway()
        result=gateway.execute_request(request_fixture(),role="Upstream",arm=ARMS[0])
        self.assertTrue(result["ok"])
        self.assertEqual(result["parsed_json"],{"fixture":True})
        self.assertAlmostEqual(result["cost_USD"],.0146)
        saved=Path(result["archive_path"])
        self.assertEqual(json.loads((saved/"request.json").read_text()),build_payload(request_fixture(),"gpt-6-astra"))
        audit=CampaignGateway(self.root,config_fixture()).audit()
        self.assertEqual(audit["attempt_count"],1)
        self.assertEqual(audit["per_arm_attempts"][ARMS[0]],1)
        self.assertAlmostEqual(audit["accounted_cost_USD"],.0146)
        for file in self.root.rglob("*"):
            if file.is_file():self.assertNotIn(FAKE_KEY.encode(),file.read_bytes())

    def test_success_without_cache_breakdown_keeps_full_reservation(self):
        def no_details(*args,**kwargs):
            status,headers,raw=mock_response(*args,**kwargs)
            data=json.loads(raw);del data["usage"]["input_tokens_details"]["cache_write_tokens"]
            return status,headers,canonical(data).encode()
        self.transport.side_effect=no_details
        result=self.gateway().execute_request(request_fixture(),role="Chief",arm=ARMS[0])
        self.assertTrue(result["ok"])
        self.assertIsNone(result["cost_USD"])
        self.assertEqual(result["accounted_cost_USD"],.32)

    def test_failed_dispatch_and_incomplete_output_are_not_retried(self):
        gateway=self.gateway()
        self.transport.side_effect=urllib.error.URLError("synthetic timeout")
        result=gateway.execute_request(request_fixture(),role="Upstream",arm=ARMS[0])
        self.assertFalse(result["ok"])
        self.assertEqual(result["accounted_cost_USD"],1.6)
        self.assertEqual(self.transport.call_count,1)
        with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(),role="Upstream",arm=ARMS[0])
        self.assertEqual(self.transport.call_count,1)
        def incomplete(*args,**kwargs):
            status,headers,raw=mock_response(*args,**kwargs)
            data=json.loads(raw);data["status"]="incomplete"
            return status,headers,canonical(data).encode()
        self.transport.side_effect=incomplete
        result=gateway.execute_request(request_fixture(),role="Chemistry",arm=ARMS[0])
        self.assertFalse(result["ok"])
        self.assertAlmostEqual(result["accounted_cost_USD"],.32)
        self.assertEqual(gateway.audit()["attempt_count"],2)

    def test_invalid_json_and_http_failure_raw_responses_are_retained(self):
        gateway=self.gateway()
        self.transport.return_value=(200,{},b"not JSON\n")
        self.transport.side_effect=None
        result=gateway.execute_request(request_fixture(),role="Chemistry",arm=ARMS[0])
        self.assertFalse(result["ok"])
        self.assertEqual((Path(result["archive_path"])/"response.raw").read_bytes(),b"not JSON\n")
        self.transport.side_effect=urllib.error.HTTPError(ENDPOINT,429,"rate limit",{},io.BytesIO(b'{"error":"synthetic"}'))
        result=gateway.execute_request(request_fixture(),role="Safety",arm=ARMS[0])
        self.assertFalse(result["ok"])
        self.assertEqual(result["response_status_code"],429)
        self.assertAlmostEqual(gateway.audit()["accounted_cost_USD"],.64)

    def test_model_tier_and_usage_mismatch_halt_future_dispatch(self):
        for field,value in [("model","unauthorized-model"),("service_tier","priority"),("oversized_input",64001)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                def changed(*args,**kwargs):
                    status,headers,raw=mock_response(*args,**kwargs)
                    data=json.loads(raw)
                    if field=="oversized_input":data["usage"]["input_tokens"]=value
                    else:data[field]=value
                    return status,headers,canonical(data).encode()
                self.transport.side_effect=changed
                gateway=CampaignGateway(Path(temp)/"campaign",config_fixture())
                result=gateway.execute_request(request_fixture(),role="Chemistry",arm=ARMS[0])
                self.assertFalse(result["ok"])
                self.assertTrue(result["halt_campaign"])
                with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(),role="Safety",arm=ARMS[0])

    def test_budget_reserved_before_dispatch_and_arm_limits_persist(self):
        gateway=self.gateway(maximum_cost_USD=.3)
        with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(),role="Chief",arm=ARMS[0])
        self.transport.assert_not_called()
        self.assertEqual(gateway.audit()["attempt_count"],0)
        with tempfile.TemporaryDirectory() as temp:
            gateway=CampaignGateway(Path(temp)/"campaign",config_fixture(max_attempts_per_arm=1))
            gateway.execute_request(request_fixture(),role="Chemistry",arm=ARMS[0])
            with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(),role="Safety",arm=ARMS[0])
            gateway.execute_request(request_fixture(),role="Safety",arm=ARMS[1])
            self.assertEqual(gateway.audit()["attempt_count"],2)

    def test_all_fourteen_unique_role_arm_attempts_and_no_spares(self):
        gateway=self.gateway()
        for arm in ARMS:
            for role in ("Upstream",)+COUNCIL_ROLES:
                self.assertTrue(gateway.execute_request(request_fixture(),role=role,arm=arm)["ok"])
        self.assertEqual(gateway.audit()["attempt_count"],14)
        self.assertEqual(self.transport.call_count,14)
        with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(),role="Chief",arm=ARMS[1])
        self.assertEqual(self.transport.call_count,14)

    def test_archive_tamper_or_deleted_latest_ledger_fails_closed(self):
        gateway=self.gateway()
        result=gateway.execute_request(request_fixture(),role="Upstream",arm=ARMS[0])
        (Path(result["archive_path"])/"request.json").write_text("changed")
        with self.assertRaises(GatewayError):gateway.audit()
        with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(),role="Safety",arm=ARMS[0])
        with tempfile.TemporaryDirectory() as temp:
            gateway=CampaignGateway(Path(temp)/"campaign",config_fixture())
            gateway.execute_request(request_fixture(),role="Safety",arm=ARMS[0])
            sorted((gateway.root/"ledger").iterdir())[-1].unlink()
            with self.assertRaises(GatewayError):gateway.audit()

    def test_lock_rejects_global_parallel_and_configuration_changes(self):
        gateway=self.gateway()
        with gateway._lock():
            with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(),role="Chief",arm=ARMS[0])
            with self.assertRaises(GatewayError):CampaignGateway(self.root,config_fixture())
        with self.assertRaises(GatewayError):CampaignGateway(self.root,config_fixture(maximum_cost_USD=20))
        self.transport.assert_not_called()

    def test_missing_key_or_credential_in_input_never_dispatches_or_saves_key(self):
        gateway=self.gateway()
        with patch.dict(os.environ,{"OPENAI_API_KEY":""}):
            with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(),role="Chief",arm=ARMS[0])
        with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(user_prompt=FAKE_KEY),role="Chief",arm=ARMS[0])
        self.transport.assert_not_called()
        for file in self.root.rglob("*"):
            if file.is_file():self.assertNotIn(FAKE_KEY.encode(),file.read_bytes())

    def test_large_input_uses_exact_count_with_schema_and_archives_it(self):
        gateway=self.gateway()
        schema={"type":"object","properties":{"fixture":{"type":"boolean"}},"required":["fixture"],"additionalProperties":False}
        result=gateway.execute_request(request_fixture(user_prompt="fixture "*12000,response_schema=schema),role="Chief",arm=ARMS[0])
        self.assertTrue(result["ok"])
        self.assertEqual(self.transport.call_count,2)
        count_call=self.transport.call_args_list[0]
        self.assertEqual(count_call.kwargs["endpoint"],TOKEN_COUNT_ENDPOINT)
        self.assertEqual(json.loads(count_call.args[0])["text"]["format"]["schema"],schema)
        audit=gateway.audit()
        self.assertEqual(audit["input_count_operations"],1)
        self.assertEqual(audit["attempt_count"],1)
        self.assertEqual(audit["entries"][0]["input_token_upper_bound"],21024)

    def test_excessive_or_failed_exact_count_never_generates(self):
        gateway=self.gateway()
        self.transport.side_effect=lambda *a,**kw:(200,{},b'{"object":"response.input_tokens","input_tokens":65000}')
        with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(user_prompt="fixture "*12000),role="Chief",arm=ARMS[0])
        self.assertEqual(gateway.audit()["attempt_count"],0)
        self.assertEqual(self.transport.call_count,1)
        self.transport.side_effect=urllib.error.URLError("synthetic count failure")
        with self.assertRaises(GatewayError):gateway.execute_request(request_fixture(user_prompt="fixture "*12000),role="Chief",arm=ARMS[0])
        self.assertEqual(gateway.audit()["attempt_count"],0)
        self.assertEqual(gateway.audit()["input_count_operations"],2)


class HTTPContractTests(unittest.TestCase):
    def test_verified_ssl_exact_endpoint_and_no_secret_in_body(self):
        response=unittest.mock.MagicMock()
        response.__enter__.return_value=response
        response.geturl.return_value=ENDPOINT
        response.status=200;response.headers={};response.read.return_value=b"{}"
        opener=unittest.mock.MagicMock();opener.open.return_value=response
        with patch("khu_sulfoxidation.model_gateway.urllib.request.build_opener",return_value=opener) as build:
            _http_post(b'{"fixture":true}',FAKE_KEY,60)
            context=build.call_args.args[0]._context
            self.assertEqual(context.verify_mode,ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            request=opener.open.call_args.args[0]
            self.assertEqual(request.full_url,ENDPOINT)
            self.assertEqual(request.get_method(),"POST")
            self.assertEqual(request.get_header("Authorization"),"Bearer "+FAKE_KEY)
            self.assertNotIn(FAKE_KEY.encode(),request.data)
        with self.assertRaises(GatewayError):_http_post(b"{}",FAKE_KEY,60,endpoint="http://api.openai.com/v1/responses")


if __name__=="__main__":unittest.main()
