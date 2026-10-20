#!/usr/bin/env python3
"""Check the full software release in the Git index, keeping local study data private.

The index is the exact snapshot Git would commit. Untracked or ignored research
files are intentionally not scanned, moved, or removed. Diagnostics contain
paths and line numbers, never matched credential values.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

MANIFEST = 'RELEASE_MANIFEST.json'
MANIFEST_SCHEMA = 'flowpilotx-full-source-release-v2'
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
ROOT_FILES = {
    '.gitignore', '.gitattributes', '.env.example', 'README.md', 'LICENSE_STATUS.md',
    'LICENSE', 'AGENTS.md', 'pyproject.toml', MANIFEST, 'requirements-runtime.txt',
    'requirements-study.txt', 'requirements.txt', 'pytest.ini', 'CHANGELOG.md',
    'requirements-optional.txt', 'requirements-runtime-constraints.txt',
}
ROOT_DIRS = {'baseline', 'src', 'tests', 'docs', 'examples', 'scripts', 'tools', '.github'}
PRIVATE_PARTS = {
    'data', 'private', 'configs', 'outputs', 'review', 'provenance', 'tmp',
    'node_modules', '.venv', 'venv', '__pycache__', '.pytest_cache', '.flowpilotx',
    'build', 'dist', 'chroma_db', 'inventory_profiles', 'deliverables', 'extraction_results',
}
PRIVATE_PATHS = {
    'baseline/flowpilot/paper_miner_history.json',
    'baseline/flowpilot/main.py',  # Credential-bearing exploratory scratch file.
    'baseline/flowpilot/scripts/run_figure5_oxygen_rerun.py',
    'baseline/flowpilot/flora_translate/tests/fixtures/scientific_giese.json',
    'tests/test_history.py',
    'tests/test_history_hypotheses.py',
    'tests/test_campaign_evidence.py',
}
BINARY_ASSET_PREFIX = 'baseline/flowpilot/flora_design/visualizer/icons/'
BINARY_ASSET_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.ico'}
DOCUMENTATION_IMAGE_PREFIX = 'docs/figures/'
DOCUMENTATION_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
PRIVATE_SUFFIXES = {
    '.pptx', '.ppt', '.pdf', '.xlsx', '.xls', '.zip', '.tar', '.gz', '.7z',
    '.sqlite', '.sqlite3', '.db', '.bin', '.pkl', '.pickle', '.npy', '.npz',
    '.pem', '.key', '.pyc', '.pyo', '.log',
}
# Concrete paths prevent a small generic demo from passing as the full system.
REQUIRED_COMPONENTS = {
    'literature retrieval': [
        'baseline/flowpilot/flora_translate/retriever.py',
        'baseline/flowpilot/flora_translate/analogy_selector.py',
        'baseline/flowpilot/flora_translate/embedding_engine.py',
        'baseline/flowpilot/flora_translate/vector_store.py',
    ],
    'supplemental literature and figure tools': [
        'baseline/flowpilot/flora_scout/__init__.py',
        'baseline/flowpilot/flora_scout/scout.py',
        'baseline/flowpilot/flora_prism/__init__.py',
        'baseline/flowpilot/flora_prism/prism.py',
        'baseline/flowpilot/paper_knowledge_extractor.py',
        'baseline/flowpilot/visualization/benchmark_figure_studio.py',
    ],
    'FlowPilot translation and councils': [
        'baseline/flowpilot/flora_translate/main.py',
        'baseline/flowpilot/flora_translate/schemas.py',
        'baseline/flowpilot/flora_translate/engine/llm_agents.py',
        'baseline/flowpilot/flora_translate/engine/council_v3/chief.py',
        'baseline/flowpilot/flora_translate/engine/council_v4/chief.py',
        'baseline/flowpilot/flora_translate/engine/council_v4/scientific.py',
    ],
    'engineering and topology': [
        'baseline/flowpilot/flora_translate/final_engineering.py',
        'baseline/flowpilot/flora_translate/design_calculator.py',
        'baseline/flowpilot/flora_translate/gas_settings.py',
        'baseline/flowpilot/flora_translate/inventory_allocator.py',
        'baseline/flowpilot/flora_translate/topology_compiler.py',
        'baseline/flowpilot/flora_design/visualizer/flowsheet_builder.py',
        'baseline/flowpilot/flora_design/visualizer/icons/bpr.png',
        'baseline/flowpilot/flora_design/visualizer/icons/check_valve.png',
        'baseline/flowpilot/flora_design/visualizer/icons/mfc.png',
        'baseline/flowpilot/flora_design/visualizer/icons/coil_reactor.png',
        'baseline/flowpilot/flora_design/visualizer/icons/pump_generic.svg',
        'baseline/flowpilot/flora_design/rules/unit_op_rules.py',
    ],
    'fundamentals and optimization': [
        'baseline/flowpilot/flora_fundamentals/handbook_reader.py',
        'baseline/flowpilot/flora_fundamentals/knowledge_store.py',
        'baseline/flowpilot/flora_optimize/bo_engine.py',
    ],
    'original interfaces': [
        'baseline/flowpilot/app.py',
        'baseline/flowpilot/pages/translate.py',
        'baseline/flowpilot/flowpilot_webapp/backend/app.py',
        'baseline/flowpilot/flowpilot_webapp/frontend/src/main.tsx',
        'baseline/flowpilot/flowpilot_webapp/frontend/package.json',
    ],
    'FlowPilotX campaign and interface': [
        'src/flowpilotx/campaign.py', 'src/flowpilotx/evidence.py',
        'src/flowpilotx/inventory.py', 'src/flowpilotx/selection.py',
        'src/flowpilotx/gateway.py', 'src/flowpilotx/server.py',
        'src/flowpilotx/gui/index.html', 'src/flowpilotx/gui/app.js',
    ],
    'KHU study adapter': [
        'src/khu_sulfoxidation/history.py',
        'src/khu_sulfoxidation/history_hypotheses.py',
        'src/khu_sulfoxidation/corrected_design.py',
        'src/khu_sulfoxidation/flowpilotx_profile.py',
        'src/khu_sulfoxidation/flowpilotx_report.py',
        'src/khu_sulfoxidation/flowpilotx_astra.py',
        'src/khu_sulfoxidation/feedback.py',
    ],
}
PATTERNS = {
    'private-key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'api-key-like': re.compile(r'\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16})\b'),
    'private-network-address': re.compile(r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b'),
}
HOME_PATH = re.compile(r'/(?:home|Users)/([A-Za-z0-9_.<>-]+)/')
HOME_PLACEHOLDERS = {'user', 'username', 'your-user', 'your-username', 'example', '<user>', '<username>'}
CREDENTIAL_LITERAL = re.compile(
    r'''(?ix)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|secret)['"]?\s*[:=]\s*['"]([^'"\r\n]{8,})['"]'''
)
PLACEHOLDER = re.compile(r'(?i)(?:^|[-_\s])(?:test|fake|dummy|synthetic|example|placeholder|redacted|replace|your|not-real|not-a-real)(?:$|[-_\s])')


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(['git', '-C', str(root), *args], input=input_bytes,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        # Do not echo Git stderr: it can include source paths or credential URLs.
        raise ValueError(f'Git {args[0]} failed; check the repository and index')
    return result.stdout


def _index_files(root: Path) -> tuple[list[dict], list[dict]]:
    entries, failures = [], []
    for raw in _git(root, 'ls-files', '--stage', '-z').split(b'\0'):
        if not raw:
            continue
        header, raw_path = raw.split(b'\t', 1)
        mode, oid, stage = header.decode('ascii').split()
        path = raw_path.decode('utf-8', 'surrogateescape')
        if stage != '0':
            failures.append({'path': path, 'reason': 'unresolved Git index conflict'})
            continue
        if mode not in {'100644', '100755'}:
            failures.append({'path': path, 'reason': 'symlinks and submodules are not release source files'})
            continue
        entries.append({'path': path, 'mode': mode, 'oid': oid})
    entries.sort(key=lambda item: item['path'])
    if not entries:
        failures.append({'path': '.', 'reason': 'Git index contains no release files'})
        return entries, failures
    # Inspect object sizes before reading bodies: accidental huge staging must
    # fail without loading an archive or populated database into memory.
    request = ('\n'.join(e['oid'] for e in entries) + '\n').encode()
    headers = _git(root, 'cat-file', '--batch-check', input_bytes=request).splitlines()
    if len(headers) != len(entries):
        raise ValueError('Incomplete Git object metadata')
    for item, raw_header in zip(entries, headers):
        header = raw_header.split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError('Git index references a non-blob or unavailable object')
        item['size'] = int(header[2])
        item['content'] = None
    within_total_limit = sum(item['size'] for item in entries) <= MAX_TOTAL_BYTES
    readable = [item for item in entries if within_total_limit and item['size'] <= MAX_FILE_BYTES]
    if readable:
        # Object IDs are passed as stdin, never interpolated shell expressions.
        request = ('\n'.join(e['oid'] for e in readable) + '\n').encode()
        data = _git(root, 'cat-file', '--batch', input_bytes=request)
        position = 0
        for item in readable:
            end = data.index(b'\n', position)
            header = data[position:end].split()
            if len(header) != 3 or header[1] != b'blob' or int(header[2]) != item['size']:
                raise ValueError('Git object metadata changed during inspection')
            size = item['size']
            position = end + 1
            item['content'] = data[position:position + size]
            if len(item['content']) != size or data[position + size:position + size + 1] != b'\n':
                raise ValueError('Incomplete Git object read')
            position += size + 1
    return entries, failures


def _content_failures(path: str, content: str) -> list[dict]:
    failures = []
    documentation = PurePosixPath(path).suffix.lower() in {'.md', '.rst', '.txt'} or path == '.env.example'
    for number, line in enumerate(content.splitlines(), 1):
        reasons = {label for label, regex in PATTERNS.items() if regex.search(line)}
        for match in HOME_PATH.finditer(line):
            # Exempt generic documentation examples, never real usernames.
            if not (documentation and match.group(1).lower() in HOME_PLACEHOLDERS):
                reasons.add('absolute-home-path')
        for match in CREDENTIAL_LITERAL.finditer(line):
            value = match.group(1)
            if not PLACEHOLDER.search(value) and value not in {'ollama', 'local-only', 'not-needed'}:
                reasons.add('credential-literal')
        failures.extend({'path': path, 'line': number, 'reason': reason} for reason in sorted(reasons))
    return failures


def _inspect(root: Path, verify_manifest: bool = False) -> tuple[dict, list[dict]]:
    try:
        entries, failures = _index_files(root)
    except (OSError, ValueError, UnicodeError) as exc:
        return {'status': 'failed', 'file_count': 0, 'total_bytes': 0,
                'failures': [{'path': '.', 'reason': str(exc)}]}, []
    records, total = [], 0
    indexed = {item['path']: item for item in entries}
    for item in entries:
        path, data = item['path'], item['content']
        rel = PurePosixPath(path)
        if (len(rel.parts) == 1 and path not in ROOT_FILES) or (len(rel.parts) > 1 and rel.parts[0] not in ROOT_DIRS):
            failures.append({'path': path, 'reason': 'unexpected release location'})
        if any(part in PRIVATE_PARTS or part.endswith('.egg-info') for part in rel.parts) or path in PRIVATE_PATHS:
            failures.append({'path': path, 'reason': 'private data, generated state, or identified private source file'})
        if (rel.name.startswith('.env') and rel.name != '.env.example') or rel.name == 'secrets.toml' or rel.suffix.lower() in PRIVATE_SUFFIXES:
            failures.append({'path': path, 'reason': 'credential, data-store, or generated artifact type excluded'})
        if item['size'] > MAX_FILE_BYTES:
            failures.append({'path': path, 'reason': 'file exceeds 10 MiB release limit'})
        total += item['size']
        if data is None:
            continue  # Body reads are intentionally skipped for oversized snapshots.
        allowed_binary = (
            path.startswith(BINARY_ASSET_PREFIX) and rel.suffix.lower() in BINARY_ASSET_SUFFIXES
        ) or (
            path.startswith(DOCUMENTATION_IMAGE_PREFIX)
            and rel.suffix.lower() in DOCUMENTATION_IMAGE_SUFFIXES
        )
        if not allowed_binary:
            try:
                content = data.decode('utf-8')
                if '\0' in content:
                    raise UnicodeError('NUL in source')
            except UnicodeError:
                failures.append({'path': path, 'reason': 'unexpected binary outside diagram assets'})
            else:
                failures.extend(_content_failures(path, content))
        if path != MANIFEST:
            records.append({'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'mode': item['mode']})
    for component, paths in REQUIRED_COMPONENTS.items():
        for path in paths:
            if path not in indexed:
                failures.append({'path': path, 'reason': f'missing required component: {component}'})
    if total > MAX_TOTAL_BYTES:
        failures.append({'path': '.', 'reason': 'release exceeds 100 MiB total limit'})
    if verify_manifest:
        if MANIFEST not in indexed:
            failures.append({'path': MANIFEST, 'reason': 'manifest must be staged before verification'})
        else:
            try:
                manifest = json.loads(indexed[MANIFEST]['content'])
                if manifest.get('schema_version') != MANIFEST_SCHEMA:
                    raise ValueError('manifest schema does not describe the full source release')
                expected_rows = manifest['files']
                expected = {row['path']: row for row in expected_rows}
                if len(expected) != len(expected_rows):
                    raise ValueError('manifest contains duplicate paths')
                actual = {row['path']: row for row in records}
                for path in sorted(expected.keys() | actual.keys()):
                    if expected.get(path) != actual.get(path):
                        failures.append({'path': path, 'reason': 'manifest file, size, mode, or hash differs from Git index'})
            except (ValueError, KeyError, TypeError, AttributeError, UnicodeError):
                failures.append({'path': MANIFEST, 'reason': 'invalid or outdated full-source manifest'})
    return {
        'status': 'passed' if not failures else 'failed',
        'file_count': len(entries), 'total_bytes': total,
        'required_component_groups': len(REQUIRED_COMPONENTS),
        'failures': failures,
        'scope': 'Git index only: full software presence, private-file boundary, sizes and limited credential/address patterns. Local private files are preserved. This is not a comprehensive security or redistribution-rights guarantee.',
    }, records


def scan(root: Path, verify_manifest: bool = False) -> dict:
    """Inspect staged/tracked index bytes, never unstaged working-tree content."""
    return _inspect(Path(root).resolve(), verify_manifest)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-manifest', action='store_true', help='Compare the staged manifest with staged file bytes')
    modes.add_argument('--write-manifest', action='store_true', help='Write a manifest of the clean Git index; stage the manifest afterward')
    args = parser.parse_args()
    root = args.root.resolve()
    result, records = _inspect(root, args.verify_manifest)
    if args.write_manifest and result['status'] == 'passed':
        manifest = {
            'schema_version': MANIFEST_SCHEMA,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'release_mode': 'full_software_private_study_inputs_excluded',
            'source_of_truth': 'Git index; manifest excludes itself',
            'private_data_included': False,
            'components': REQUIRED_COMPONENTS,
            'license': 'No new redistribution grant; see LICENSE_STATUS.md',
            'files': records,
        }
        destination = root / MANIFEST
        if destination.is_symlink():
            result['failures'].append({'path': MANIFEST, 'reason': 'refusing to write through a manifest symlink'})
            result['status'] = 'failed'
        else:
            destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
            result['manifest_written'] = MANIFEST
            result['next_step'] = 'Stage RELEASE_MANIFEST.json, then run --verify-manifest.'
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
