"""Documented patches must not conceal changes to protected originals."""
import hashlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

spec = importlib.util.spec_from_file_location("verify_handoff", Path(__file__).resolve().parents[1] / "scripts/verify_handoff.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


class HandoffIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = "baseline/flowpilot/example.py"
        self.write(self.path, "original")
        self.manifest = {"files": [{"destination": self.path, "sha256": sha("original")}]}
        self.patch = {"files": [{"path": self.path, "original_sha256": sha("original"),
            "updated_sha256": sha("reviewed"), "backup": "private/original.py", "diff": "private/change.patch"}]}

    def write(self, path, value):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def test_original_and_documented_change(self):
        self.assertEqual(module.verify_source_files(self.root, self.manifest)["unchanged_source_files"], 1)
        self.write(self.path, "reviewed")
        self.write("private/original.py", "original")
        self.write("private/change.patch", "reviewed patch")
        result = module.verify_source_files(self.root, self.manifest, self.patch)
        self.assertEqual(result["documented_patched_files"], 1)
        self.assertEqual(result["unchanged_source_files"], 0)

    def test_unrecorded_or_further_changes_fail(self):
        self.write(self.path, "reviewed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            module.verify_source_files(self.root, self.manifest)
        self.write("private/original.py", "original")
        self.write("private/change.patch", "reviewed patch")
        self.write(self.path, "unexpected")
        with self.assertRaisesRegex(ValueError, "integrity"):
            module.verify_source_files(self.root, self.manifest, self.patch)

    def test_bad_backup_or_undocumented_patch_fail(self):
        self.write(self.path, "reviewed")
        self.write("private/original.py", "tampered")
        self.write("private/change.patch", "reviewed patch")
        with self.assertRaisesRegex(ValueError, "integrity"):
            module.verify_source_files(self.root, self.manifest, self.patch)
        self.write("private/original.py", "original")
        (self.root / "private/change.patch").unlink()
        with self.assertRaisesRegex(ValueError, "integrity"):
            module.verify_source_files(self.root, self.manifest, self.patch)


if __name__ == "__main__":
    unittest.main()
