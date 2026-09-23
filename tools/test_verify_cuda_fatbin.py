"""Fail-closed CUDA image audit tests without a GPU or CUDA Toolkit."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import importlib.util


VERIFY = Path(__file__).with_name("verify_cuda_fatbin.py")
SPEC = importlib.util.spec_from_file_location("cuda_fatbin_audit", VERIFY)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@unittest.skipIf(os.name == "nt", "POSIX fake cuobjdump executable")
class ImageAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binaries = [self.root / "vrhino", self.root / "vrhino-native"]
        for binary in self.binaries:
            binary.write_bytes(b"\x7fELF")
        self.cuobjdump = self.root / "fake-cuobjdump"
        self.cuobjdump.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            "images = json.loads(os.environ['FAKE_CUDA_IMAGES'])\n"
            "if os.environ.get('FAKE_CUDA_DUMP_FAIL') == '1':\n"
            "    print('synthetic cuobjdump failure', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "kind = 'cubin' if sys.argv[1] == '--list-elf' else 'ptx'\n"
            "for sm in images[pathlib.Path(sys.argv[2]).name][kind]:\n"
            "    print(f'Fatbin {kind}: .sm_{sm}.{kind}')\n"
        )
        self.cuobjdump.chmod(0o755)
        self.images = {
            binary.name: {"cubin": [80, 86, 89, 90, 120], "ptx": [80]}
            for binary in self.binaries
        }

    def audit(self, *, fail=False):
        env = os.environ.copy()
        env["FAKE_CUDA_IMAGES"] = json.dumps(self.images)
        if fail:
            env["FAKE_CUDA_DUMP_FAIL"] = "1"
        return subprocess.run(
            [sys.executable, str(VERIFY), *map(str, self.binaries),
             "--cuobjdump", str(self.cuobjdump)],
            text=True, capture_output=True, env=env, check=False,
        )

    def test_both_executables_have_real_images_and_ptx(self):
        result = self.audit()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA_IMAGE_AUDIT=PASS", result.stdout)
        self.assertEqual(result.stdout.count("CUBIN_SM="), 2)

    def test_second_executable_missing_image_is_rejected(self):
        self.images["vrhino-native"]["cubin"].remove(90)
        result = self.audit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing_cubin=[90]", result.stderr)

    def test_missing_ptx_is_rejected(self):
        self.images["vrhino"]["ptx"].clear()
        result = self.audit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing_ptx=True", result.stderr)

    def test_cuobjdump_failure_is_rejected(self):
        result = self.audit(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("synthetic cuobjdump failure", result.stderr)

    def test_wrong_image_type_cannot_satisfy_requirement(self):
        self.images["vrhino"]["cubin"] = [80, 86, 89, 90]
        self.images["vrhino"]["ptx"] = [80, 120]
        result = self.audit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing_cubin=[120]", result.stderr)


class PortableParsingTests(unittest.TestCase):
    def test_cubin_and_ptx_listings_remain_distinct(self):
        listing = ".sm_80.cubin\n.sm_89.cubin\n.sm_120.ptx\n"
        with mock.patch.object(AUDIT.subprocess, "run", return_value=
                               subprocess.CompletedProcess([], 0, listing, "")):
            self.assertEqual(AUDIT.images("cuobjdump", "--list-elf", Path("vrhino.exe")),
                             {80, 89})
            self.assertEqual(AUDIT.images("cuobjdump", "--list-ptx", Path("vrhino.exe")),
                             {120})

    def test_cuobjdump_error_is_not_treated_as_empty_listing(self):
        with mock.patch.object(AUDIT.subprocess, "run", return_value=
                               subprocess.CompletedProcess([], 1, "", "bad PE image")):
            with self.assertRaisesRegex(RuntimeError, "bad PE image"):
                AUDIT.images("cuobjdump", "--list-elf", Path("vrhino.exe"))

    def test_missing_cuobjdump_is_reported_as_audit_failure(self):
        with mock.patch.object(AUDIT.subprocess, "run", side_effect=FileNotFoundError("missing")):
            with self.assertRaisesRegex(RuntimeError, "cannot run cuobjdump"):
                AUDIT.images("cuobjdump", "--list-elf", Path("vrhino.exe"))


if __name__ == "__main__":
    unittest.main()
