"""Read-only replay regression on retained Responses attempts; no gateway lock/calls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from khu_sulfoxidation.flowpilotx_astra import request_for, _cached, ROLES, SPECIALISTS, ARM
from khu_sulfoxidation.flowpilotx_astra_council import ordered_response

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/flowpilotx_astra_sol_2026-10-15_v1'


def read(path):
    return json.loads(path.read_text())


class AstraReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = read(RUN / 'input_freeze.json')
        cls.frozen = cls.freeze['frozen_inputs']
        cls.sources = cls.freeze['configuration']['source_hashes']
        cls.upstream = read(RUN / 'api/attempts/attempt-0001/result.json')
        cls.chemistry = read(RUN / 'api/attempts/attempt-0002/result.json')

    def test_original_text_restores_exact_previously_dispatched_chemistry_request(self):
        actual = read(RUN / 'api/attempts/attempt-0002/input.json')
        restored = request_for(self.frozen, 'Chemistry', {'Upstream': ordered_response(self.upstream)}, self.sources)
        sorted_object = request_for(self.frozen, 'Chemistry', {'Upstream': self.upstream['parsed_json']}, self.sources)
        self.assertEqual(restored, actual)
        self.assertEqual(restored['user_prompt'].encode(), actual['user_prompt'].encode())
        self.assertNotEqual(sorted_object['user_prompt'], actual['user_prompt'])
        self.assertEqual(json.loads(sorted_object['user_prompt']), json.loads(actual['user_prompt']))
        gateway = SimpleNamespace(root=RUN / 'api', audit=lambda: {'entries': [
            {'role': 'Chemistry', 'arm': ARM, 'attempt_id': 'attempt-0002', 'status': 'completed'}]})
        self.assertEqual(_cached(gateway, 'Chemistry', restored), self.chemistry)
        with self.assertRaisesRegex(ValueError, 'differs from frozen request'):
            _cached(gateway, 'Chemistry', sorted_object)

    def test_cached_and_fresh_serialization_preserve_same_upcoming_context(self):
        upstream = ordered_response(self.upstream)
        chemistry = ordered_response(self.chemistry)
        fresh = {'Upstream': upstream, 'Chemistry': chemistry}
        cached = {role: ordered_response(json.loads(json.dumps({'text': json.dumps(response, ensure_ascii=False),
                    'parsed_json': response}, sort_keys=True))) for role, response in fresh.items()}
        for role in SPECIALISTS[1:]:
            direct = request_for(self.frozen, role, fresh, self.sources)
            replay = request_for(self.frozen, role, cached, self.sources)
            self.assertEqual(direct, replay)
            context = json.loads(replay['user_prompt'])
            self.assertEqual(context.pop('upstream_proposal'), upstream)
            self.assertEqual(context.pop('assigned_role'), role)
            self.assertEqual(context, self.frozen['context'])
        # Synthetic placeholders exercise only prospective assembly, not science.
        for role in SPECIALISTS[1:]:
            fresh[role] = {'role': role, 'synthetic_fixture': True, 'issues': [{'z': 'last', 'a': 'first'}]}
            cached[role] = ordered_response({'text': json.dumps(fresh[role]), 'parsed_json': deepcopy(fresh[role])})
        fresh['Skeptic'] = {'role': 'Skeptic', 'synthetic_fixture': True, 'issues': []}
        cached['Skeptic'] = deepcopy(fresh['Skeptic'])
        for role in ['Skeptic', 'Chief']:
            request = request_for(self.frozen, role, fresh, self.sources)
            self.assertEqual(request, request_for(self.frozen, role, cached, self.sources))
            context = json.loads(request['user_prompt'])
            for key, value in self.frozen['context'].items():
                self.assertEqual(context[key], value)
            self.assertEqual([x['role'] for x in context['specialist_reviews']], list(SPECIALISTS))
            self.assertEqual(context['upstream_proposal'], upstream)

    def test_text_and_decoded_object_disagreement_or_nonobject_rejected(self):
        for answer in [{'text': '{"a":1}', 'parsed_json': {'a': 2}},
                       {'text': '[]', 'parsed_json': []}, {'text': 'null', 'parsed_json': None}]:
            with self.assertRaisesRegex(ValueError, 'text/object mismatch'):
                ordered_response(answer)
        with self.assertRaises(json.JSONDecodeError):
            ordered_response({'text': '{', 'parsed_json': {}})


if __name__ == '__main__': unittest.main()
