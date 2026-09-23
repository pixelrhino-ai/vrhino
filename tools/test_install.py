#!/usr/bin/env python3
"""Offline installer integration tests; never download or execute model code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PIN = '37bff0ecd0ac10d3bf2bd7c2c78f867477ca27a3c803d83f32a6d873ce015733'

class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "user's home"
        self.home.mkdir()
        self.prefix = self.home / 'runtime'
        self.bin = self.home / 'commands'
        tree = self.root / 'vrhino-v0.9.1-alpha'
        (tree / 'bin').mkdir(parents=True)
        for name in ('vrhino', 'vrhino-wan-family-convert'):
            p = tree / 'bin' / name
            p.write_text('#!/bin/sh\nif [ "${1:-}" = --version ]; then echo "VRhino v0.9.1-alpha"; else printf "%s\\n" "$@"; fi\n')
            p.chmod(0o755)
        (tree / 'SHA256SUMS').write_text(''.join(
            hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(tree)) + '\n'
            for p in sorted((tree / 'bin').iterdir())))
        self.archive = self.root / 'fixture.tar.gz'
        with tarfile.open(self.archive, 'w:gz') as t:
            t.add(tree, arcname=tree.name)
        self.script = self.root / 'install.sh'
        # Only the temporary test copy accepts the tiny archive. The shipped script
        # has no digest/URL bypass; its pin is checked independently below.
        self.script.write_text((ROOT / 'install.sh').read_text().replace(PIN, hashlib.sha256(self.archive.read_bytes()).hexdigest()))
        self.env = dict(os.environ, HOME=str(self.home), SHELL='/bin/bash')
        self.env.pop('VRHINO_INSTALL_DIR', None)
        self.env.pop('VRHINO_BIN_DIR', None)
        self.env.pop('XDG_DATA_HOME', None)

    def install(self, *extra, script=None):
        return subprocess.run(['sh', str(script or self.script), '--prefix', str(self.prefix),
            '--bin-dir', str(self.bin), '--archive', str(self.archive), *extra],
            env=self.env, text=True, capture_output=True)

    def test_release_pin_matches_public_record(self):
        record = json.loads((ROOT / 'docs/release/v0.9.1-alpha-assets.json').read_text())
        self.assertEqual(next(a['sha256'] for a in record['assets'] if a['name'].startswith('vrhino-linux') and a['name'].endswith('.tar.gz')), PIN)
        self.assertIn('archive_sha='+PIN, (ROOT / 'install.sh').read_text())

    def test_install_spaces_quotes_arguments_and_path(self):
        r = self.install(); self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        out = subprocess.check_output([str(self.bin/'vrhino'), 'one argument', 'two'], text=True)
        self.assertEqual(out, 'one argument\ntwo\n')
        line = next(x for x in (self.home/'.profile').read_text().splitlines() if x.startswith('export PATH='))
        r = subprocess.run(['sh', '-c', line+'\ncommand -v vrhino'], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), str(self.bin/'vrhino'))

    def test_repeat_does_not_reextract_or_duplicate_profile(self):
        self.assertEqual(self.install().returncode, 0)
        stamp = (self.prefix/'bin/vrhino').stat().st_mtime_ns
        before = (self.home/'.profile').read_text()
        r = self.install(); self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(stamp, (self.prefix/'bin/vrhino').stat().st_mtime_ns)
        self.assertEqual(before, (self.home/'.profile').read_text())

    def test_corrupt_archive_rejected_before_extract(self):
        self.archive.write_bytes(b'bad')
        r = self.install(); self.assertNotEqual(r.returncode, 0)
        self.assertFalse(self.prefix.exists())
        self.assertFalse(list(self.home.glob('.vrhino-install.*')))

    def test_unmanaged_directory_preserved(self):
        self.prefix.mkdir(); (self.prefix/'keep').write_text('keep')
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual((self.prefix/'keep').read_text(), 'keep')

    def test_existing_command_preserved(self):
        self.bin.mkdir(); (self.bin/'vrhino').write_text('existing')
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual((self.bin/'vrhino').read_text(), 'existing')
        self.assertFalse(self.prefix.exists())

    def test_existing_install_corruption_rejected(self):
        self.assertEqual(self.install().returncode, 0)
        (self.prefix/'bin/vrhino').write_text('bad')
        self.assertNotEqual(self.install().returncode, 0)

    def test_no_profile_modification(self):
        r = self.install('--no-modify-path'); self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.home/'.profile').exists())
        self.assertFalse((self.home/'.bashrc').exists())

    def test_lock_preserved(self):
        lock = Path(str(self.prefix)+'.install-lock'); lock.mkdir()
        self.assertNotEqual(self.install().returncode, 0)
        self.assertTrue(lock.exists())

    def test_symlink_prefix_rejected(self):
        target = self.home/'other'; target.mkdir(); self.prefix.symlink_to(target)
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(list(target.iterdir()), [])

    def test_pipe_install(self):
        p = subprocess.run(['sh', '-s', '--', '--prefix', str(self.prefix), '--bin-dir', str(self.bin),
                            '--archive', str(self.archive), '--no-modify-path'],
                           input=self.script.read_text(), env=self.env, text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        self.assertTrue((self.bin/'vrhino').is_file())

    def test_pipe_help(self):
        p = subprocess.run(['sh', '-s', '--', '--help'], input=(ROOT/'install.sh').read_text(), text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('Usage:', p.stdout)

    def test_unsupported_platform_rejected(self):
        mock = self.root/'platform'; mock.mkdir()
        uname = mock/'uname'; uname.write_text('#!/bin/sh\necho Darwin\n'); uname.chmod(0o755)
        self.env['PATH'] = str(mock)+':'+self.env['PATH']
        self.assertNotEqual(self.install().returncode, 0)
        self.assertFalse(self.prefix.exists())

    def test_old_glibc_rejected(self):
        mock = self.root/'glibc'; mock.mkdir()
        tool = mock/'getconf'; tool.write_text('#!/bin/sh\necho "glibc 2.31"\n'); tool.chmod(0o755)
        self.env['PATH'] = str(mock)+':'+self.env['PATH']
        self.assertNotEqual(self.install().returncode, 0)
        self.assertFalse(self.prefix.exists())

    def test_unset_shell_supported(self):
        self.env.pop('SHELL', None)
        r = self.install(); self.assertEqual(r.returncode, 0, r.stderr)

    def test_download_failure_cleans_staging(self):
        mock = self.root/'mock'; mock.mkdir()
        curl = mock/'curl'; curl.write_text('#!/bin/sh\nexit 22\n'); curl.chmod(0o755)
        env = dict(self.env, PATH=str(mock)+':'+self.env['PATH'])
        r = subprocess.run(['sh', str(self.script), '--prefix', str(self.prefix), '--bin-dir', str(self.bin)], env=env, capture_output=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(self.prefix.exists())
        self.assertFalse(list(self.home.glob('.vrhino-install.*')))

if __name__ == '__main__':
    unittest.main(verbosity=2)
