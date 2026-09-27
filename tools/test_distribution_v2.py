#!/usr/bin/env python3
"""Failure and reuse tests for the two-artifact offline installer."""
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('install_distribution_v2.sh')
ROLLBACK = Path(__file__).with_name('rollback_distribution_v2_legacy.sh')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(base, root, name, files):
    tree = base / root
    tree.mkdir()
    for rel, data in files.items():
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        if rel.startswith('bin/'):
            p.chmod(0o755)
    (tree / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(tree)}\n'
        for p in sorted(tree.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS', 'SYMLINKS.tsv')))
    (tree / 'SYMLINKS.tsv').write_text('')
    path = base / name
    with tarfile.open(path, 'w:gz') as t:
        t.add(tree, arcname=root)
    return path, digest(path)


class DistributionV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.bundle = self.base / 'bundle'
        self.bundle.mkdir()
        self.prefix = self.base / 'store'
        self.bin = self.base / 'bin'
        self.runtime_id = 'nvidia-linux-x86_64-fixture-r1'
        self.runtime_root = 'runtime-fixture'
        self.runtime_file, self.runtime_sha = archive(self.bundle, self.runtime_root,
            'runtime.tar.gz', {'dependency.txt': f'id={self.runtime_id}\nruntime_api=1\n'.encode(),
                               'lib/libcudart.so.12': b'fixture'})
        self.env = dict(os.environ, HOME=str(self.base))
        self.make_core('a')

    def make_core(self, version, doctor_ok=True, requirement=None):
        root = f'core-{version}'
        req = requirement or self.runtime_id
        runner = ('#!/bin/sh\n'
                  'case "$1" in --version) echo "VRhino fixture";; '
                  f'doctor) exit {0 if doctor_ok else 1};; *) echo ok;; esac\n').encode()
        self.core_file, self.core_sha = archive(self.bundle, root, f'core-{version}.tar.gz',
            {'bin/vrhino': runner, 'bin/vrhino-wan-family-convert': runner,
             'requires-nvidia.txt': f'id={req}\nsha256={self.runtime_sha}\nruntime_api=1\n'.encode()})
        self.bundle.joinpath('catalog.txt').write_text(
            'schema=vrhino-distribution-v2-catalog-v1\n'
            f'catalog_version=fixture-{version}\nplatform=linux-x86_64\nruntime_api=1\n'
            f'core_name=core-{version}.tar.gz\ncore_sha256={self.core_sha}\ncore_root={root}\n'
            f'core_version={version}\nruntime_name=runtime.tar.gz\n'
            f'runtime_sha256={self.runtime_sha}\nruntime_root={self.runtime_root}\n'
            f'runtime_id={self.runtime_id}\n')

    def install(self, *extra_args):
        return subprocess.run(['sh', str(SCRIPT), '--bundle-dir', str(self.bundle),
                               '--prefix', str(self.prefix), '--bin-dir', str(self.bin),
                               *extra_args],
                              env=self.env, capture_output=True, text=True)

    def active(self):
        return os.readlink(self.prefix / 'current')

    def make_legacy_install(self):
        old = self.base / 'v0.9.1-monolithic'
        (old / 'bin').mkdir(parents=True)
        self.bin.mkdir(exist_ok=True)
        for name in ('vrhino', 'vrhino-wan-family-convert'):
            target = old / 'bin' / name
            target.write_text('#!/bin/sh\necho "VRhino v0.9.1-alpha legacy"\n')
            target.chmod(0o755)
            launcher = self.bin / name
            launcher.write_text('#!/bin/sh\n# VRhino installer launcher v1\n'
                                f'exec "{target}" "$@"\n')
            launcher.chmod(0o755)
        return {name: (self.bin / name).read_bytes()
                for name in ('vrhino', 'vrhino-wan-family-convert')}

    def test_clean_offline_install_and_reuse(self):
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.active(), 'runtime/a')
        self.runtime_file.unlink()
        self.make_core('b')
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('Reusing NVIDIA runtime', p.stdout)
        self.assertEqual(self.active(), 'runtime/b')

    def test_missing_runtime(self):
        self.runtime_file.unlink()
        self.assertNotEqual(self.install().returncode, 0)
        self.assertFalse((self.prefix / 'current').exists())

    def test_corrupt_runtime(self):
        self.runtime_file.write_bytes(b'corrupt')
        self.assertNotEqual(self.install().returncode, 0)
        self.assertFalse((self.prefix / 'current').exists())

    def test_corrupt_core_keeps_a(self):
        self.assertEqual(self.install().returncode, 0)
        self.make_core('b')
        self.core_file.write_bytes(b'corrupt')
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(self.active(), 'runtime/a')

    def test_internal_checksum_failure(self):
        tree = self.bundle / 'core-a'
        (tree / 'bin/vrhino').write_text('corrupt after checksums')
        with tarfile.open(self.core_file, 'w:gz') as t:
            t.add(tree, arcname='core-a')
        catalog = self.bundle / 'catalog.txt'
        catalog.write_text(catalog.read_text().replace(self.core_sha, digest(self.core_file)))
        self.assertNotEqual(self.install().returncode, 0)
        self.assertFalse((self.prefix / 'current').exists())

    def test_wrong_runtime_identity_keeps_a(self):
        self.assertEqual(self.install().returncode, 0)
        self.make_core('b', requirement='wrong-runtime')
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(self.active(), 'runtime/a')

    def test_doctor_failure_keeps_a(self):
        self.assertEqual(self.install().returncode, 0)
        self.make_core('b', doctor_ok=False)
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(self.active(), 'runtime/a')

    def test_interrupted_core_archive_keeps_a(self):
        self.assertEqual(self.install().returncode, 0)
        self.make_core('b')
        self.core_file.write_bytes(self.core_file.read_bytes()[:40])
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(self.active(), 'runtime/a')

    def test_core_available_but_dependency_activation_fails(self):
        self.assertEqual(self.install().returncode, 0)
        self.runtime_id = 'nvidia-linux-x86_64-fixture-r2'
        self.runtime_root = 'runtime-fixture-b'
        self.runtime_file, self.runtime_sha = archive(self.bundle, self.runtime_root,
            'runtime-b.tar.gz', {'dependency.txt': f'id={self.runtime_id}\nruntime_api=1\n'.encode(),
                                 'lib/libcudart.so.12': b'new fixture'})
        self.make_core('b')
        catalog = self.bundle / 'catalog.txt'
        catalog.write_text(catalog.read_text().replace('runtime_name=runtime.tar.gz',
                                                       'runtime_name=runtime-b.tar.gz'))
        self.runtime_file.write_bytes(b'corrupt')
        self.assertTrue(self.core_file.is_file())
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(self.active(), 'runtime/a')

    def test_repeat_install(self):
        self.assertEqual(self.install().returncode, 0)
        self.assertEqual(self.install().returncode, 0)
        self.assertEqual(self.active(), 'runtime/a')

    def test_catalog_sha_pin(self):
        catalog_sha = digest(self.bundle / 'catalog.txt')
        self.assertEqual(self.install('--catalog-sha256', catalog_sha).returncode, 0)
        self.assertNotEqual(self.install('--catalog-sha256', '0' * 64).returncode, 0)
        self.assertEqual(self.active(), 'runtime/a')

    def test_catalog_missing_malformed_duplicate_and_platform_fail_closed(self):
        self.assertEqual(self.install().returncode, 0)
        catalog = self.bundle / 'catalog.txt'
        original = catalog.read_text()
        for changed in (None, original + 'malformed\n',
                        original + f'core_sha256={self.core_sha}\n',
                        original.replace('platform=linux-x86_64', 'platform=windows-x86_64'),
                        original.replace('schema=vrhino-distribution-v2-catalog-v1', 'schema=unknown')):
            if changed is None:
                catalog.unlink()
            else:
                catalog.write_text(changed)
            self.assertNotEqual(self.install().returncode, 0)
            self.assertEqual(self.active(), 'runtime/a')
        catalog.write_text(original)

    def test_catalog_wrong_archive_sha_and_stale_version_fail_closed(self):
        self.assertEqual(self.install().returncode, 0)
        catalog = self.bundle / 'catalog.txt'
        original = catalog.read_text()
        for old in (self.core_sha, self.runtime_sha):
            catalog.write_text(original.replace(old, '0' * 64))
            self.assertNotEqual(self.install().returncode, 0)
            self.assertEqual(self.active(), 'runtime/a')
        self.make_core('b')
        catalog.write_text(catalog.read_text().replace('core_version=b', 'core_version=a'))
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(self.active(), 'runtime/a')

    def test_corrupt_installed_runtime_is_not_reused(self):
        self.assertEqual(self.install().returncode, 0)
        private_lib = self.prefix / 'dependencies' / self.runtime_id / 'lib/libcudart.so.12'
        private_lib.write_bytes(b'tampered')
        self.make_core('b')
        p = self.install()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('internal checksum failed', p.stderr)
        self.assertEqual(self.active(), 'runtime/a')

    def test_install_path_with_spaces_and_quote(self):
        self.prefix = self.base / "user's runtime store"
        self.bin = self.base / "user's commands"
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        p = subprocess.run([str(self.bin / 'vrhino'), '--version'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('VRhino fixture', p.stdout)

    def test_legacy_upgrade_and_atomic_rollback(self):
        originals = self.make_legacy_install()
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.active(), 'runtime/a')
        for name, data in originals.items():
            self.assertEqual((self.prefix / 'legacy/v0.9.1-alpha/bin' / name).read_bytes(), data)
        self.assertTrue((self.base / 'v0.9.1-monolithic/bin/vrhino').exists())
        p = subprocess.run([str(self.bin / 'vrhino'), '--version'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0); self.assertIn('VRhino fixture', p.stdout)
        p = subprocess.run(['sh', str(ROLLBACK), '--prefix', str(self.prefix),
                            '--bin-dir', str(self.bin)], env=self.env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.active(), 'legacy/v0.9.1-alpha')
        p = subprocess.run([str(self.bin / 'vrhino'), '--version'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0); self.assertIn('v0.9.1-alpha legacy', p.stdout)
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.active(), 'runtime/a')

    def test_legacy_upgrade_corrupt_core_preserves_v1(self):
        originals = self.make_legacy_install()
        self.core_file.write_bytes(b'corrupt')
        p = self.install(); self.assertNotEqual(p.returncode, 0)
        for name, data in originals.items():
            self.assertEqual((self.bin / name).read_bytes(), data)
        self.assertTrue((self.base / 'v0.9.1-monolithic/bin/vrhino').exists())

    def test_legacy_upgrade_activation_failure_restores_launchers(self):
        originals = self.make_legacy_install()
        mock = self.base / 'mock-bin'
        mock.mkdir()
        mv = mock / 'mv'
        mv.write_text('#!/bin/sh\n'
                      'for arg in "$@"; do\n'
                      '  case "$arg" in */.stage.*/current)\n'
                      '    [ "$(readlink "$arg")" = runtime/a ] && exit 81;;\n'
                      '  esac\n'
                      'done\nexec /usr/bin/mv "$@"\n')
        mv.chmod(0o755)
        self.env['PATH'] = f'{mock}:/usr/bin:/bin'
        p = self.install(); self.assertNotEqual(p.returncode, 0)
        self.assertIn('atomic activation failed', p.stderr)
        for name, data in originals.items():
            self.assertEqual((self.bin / name).read_bytes(), data)
        self.assertFalse((self.prefix / 'current').exists())

    def test_interrupted_legacy_launcher_handoff_recovers(self):
        self.make_legacy_install()
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        p = subprocess.run(['sh', str(ROLLBACK), '--prefix', str(self.prefix),
                            '--bin-dir', str(self.bin)], env=self.env,
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        saved = self.prefix / 'legacy/v0.9.1-alpha/bin/vrhino-wan-family-convert'
        (self.bin / 'vrhino-wan-family-convert').write_bytes(saved.read_bytes())
        self.assertEqual(self.active(), 'legacy/v0.9.1-alpha')
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.active(), 'runtime/a')
        self.assertIn('VRhino fixture', subprocess.check_output(
            [str(self.bin / 'vrhino'), '--version'], text=True))

    def test_interrupted_initial_launcher_pair_recovers(self):
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        (self.bin / 'vrhino-wan-family-convert').unlink()
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue((self.bin / 'vrhino-wan-family-convert').is_file())
        self.assertEqual(self.active(), 'runtime/a')

    def test_dead_install_lock_recovers(self):
        lock = self.base / 'store.install-lock'
        lock.mkdir()
        (lock / 'pid').write_text('999999999\n')
        p = self.install(); self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.active(), 'runtime/a')


if __name__ == '__main__':
    unittest.main(verbosity=2)
