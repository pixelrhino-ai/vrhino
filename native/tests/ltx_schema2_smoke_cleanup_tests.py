#!/usr/bin/env python3
"""Bounded Linux observer-failure regression; no GPU or model assets required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ltx_schema2_source_product_smoke as smoke


class ObserverCleanup(unittest.TestCase):
    def test_gpu_observer_failure_reaps_native_process(self):
        with tempfile.TemporaryDirectory(prefix='ltx-smoke-cleanup-') as directory:
            root = Path(directory)
            cli = root / 'fake-native'
            cli.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(300)\n')
            cli.chmod(0o755)
            source = root / 'source'
            source.mkdir()
            package = root / 'package'
            package.mkdir()
            (package / 'vrhino-model.json').write_text(json.dumps({
                'identity': {'namespace': 'review', 'name': 'ltx-video-v0.9.1-schema2',
                             'version': '0.2.0'}, 'artifacts': []}))
            specs = root / 'specs'
            (specs / 'ltx_schema2_v1').mkdir(parents=True)
            (specs / 'ltx_schema2_v1/source-plan.json').write_text('{"artifacts": []}')
            arguments = ['smoke', '--cli', str(cli), '--source', str(source),
                         '--package', str(package), '--specs', str(specs),
                         '--work', str(root / 'work'), '--encoder', str(cli)]
            owned = []
            popen = subprocess.Popen

            def capture(*args, **kwargs):
                process = popen(*args, **kwargs)
                owned.append(process)
                return process

            try:
                with patch.object(sys, 'argv', arguments), \
                     patch.object(smoke.subprocess, 'Popen', side_effect=capture), \
                     patch.object(smoke, 'gpu', side_effect=['0 MiB', RuntimeError('observer failed')]):
                    with self.assertRaisesRegex(RuntimeError, 'observer failed'):
                        smoke.main()
                self.assertEqual(len(owned), 1)
                self.assertIsNotNone(owned[0].poll(), 'Native child escaped observer failure')
                with self.assertRaises(ProcessLookupError):
                    os.kill(owned[0].pid, 0)
            finally:
                for process in owned:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
