"""Focused malformed-PE, dependency-boundary, path and deterministic ZIP tests."""
import importlib.util
import copy
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('packager', Path(__file__).with_name('package_windows_build.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def fixture(name='KERNEL32.dll', delay=False):
    data = bytearray(2048)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 0x80)
    data[0x80:0x84] = b'PE\0\0'
    struct.pack_into('<HH', data, 0x84, 0x8664, 1)
    struct.pack_into('<H', data, 0x94, 240)
    opt = 0x98
    struct.pack_into('<H', data, opt, 0x20b)
    struct.pack_into('<Q', data, opt + 24, 0x140000000)
    struct.pack_into('<I', data, opt + 60, 512)
    struct.pack_into('<I', data, opt + 108, 16)
    index, width = (13, 32) if delay else (1, 20)
    struct.pack_into('<II', data, opt + 112 + index * 8, 0x1000, width * 2)
    struct.pack_into('<IIII', data, opt + 240 + 8, 1024, 0x1000, 1024, 512)
    if delay:
        struct.pack_into('<II', data, 512, 1, 0x1100)
    else:
        struct.pack_into('<I', data, 512 + 12, 0x1100)
    value = name.encode('ascii') + b'\0'
    data[768:768 + len(value)] = value
    return data


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def binary(self, data):
        path = self.root / 'test.exe'
        path.write_bytes(data)
        return path

    def test_direct_and_delay(self):
        for delay in (False, True):
            imports = p.pe_imports(self.binary(fixture(delay=delay)))
            self.assertEqual(imports, [{'name': 'KERNEL32.dll', 'kind': 'delay' if delay else 'direct'}])

    def test_unknown_dependency_rejected_even_if_installed(self):
        for delay in (False, True):
            with self.assertRaisesRegex(ValueError, 'unexpected/missing'):
                p.audit_pe({'vrhino.exe': self.binary(fixture('python312.dll', delay))})

    def test_truncated_and_unmapped_pe(self):
        data = fixture()
        for bad in (data[:32], data[:700], b'not a PE'):
            with self.assertRaises(ValueError):
                p.pe_imports(self.binary(bad))
        struct.pack_into('<I', data, 524, 0xffff0000)
        with self.assertRaisesRegex(ValueError, 'unmapped'):
            p.pe_imports(self.binary(data))

    def test_architecture_rejected(self):
        data = fixture()
        struct.pack_into('<H', data, 0x84, 0x14c)
        with self.assertRaisesRegex(ValueError, 'x64'):
            p.pe_imports(self.binary(data))

    def test_paths(self):
        for bad in ('../x', '/x', 'C:/x', 'a\\b', 'a//b', 'x/NUL.txt', 'x/a:stream', 'x/trailing.', './x'):
            with self.assertRaises(ValueError, msg=bad):
                p.safe_relative(bad)
        self.assertEqual(str(p.safe_relative('licenses/目录/name.txt')), 'licenses/目录/name.txt')
        with self.assertRaisesRegex(ValueError, 'missing file'):
            p.regular_file(self.root, 'missing.dll')

    def test_binary_private_roots(self):
        roots = ['Q:/checkout/source', 'R:/output/build', 'S:/SDK headers', 'T:/Users/person']
        for root in roots:
            for spelling in (root.lower(), root.upper().replace('/', '\\')):
                for encoding in ('utf-8', 'utf-16le'):
                    data = bytes(fixture()) + spelling.encode(encoding) + b'\0'
                    path = self.binary(data)
                    self.assertEqual(p.binary_private_paths(path, roots), [2048])
                    with self.assertRaisesRegex(ValueError, 'private build path'):
                        p.audit_binary_privacy({'product.exe': path}, roots)
        path = self.binary(bytes(fixture()) + b'https://example.com/source/header.h\0vrhino/native/header.h\0')
        self.assertEqual(p.binary_private_paths(path, roots), [])
        for invalid in ([], ['relative/path'], ['C:/']):
            with self.assertRaises(ValueError):
                p.binary_private_paths(path, invalid)

    def test_binary_private_root_read_boundary(self):
        root = 'Z:/private/build'
        path = self.binary(b'\0' * (1024 * 1024 - 4) + root.encode() + b'/source.cpp')
        self.assertEqual(p.binary_private_paths(path, [root, 'Z:/private']), [1024 * 1024 - 4])

    def test_allowlist_and_compiler_pair(self):
        self.assertEqual(len(p.DLLS), 39)
        self.assertEqual(len({x.lower() for x in p.DLLS}), 39)
        self.assertNotIn('nvJitLink_120_0.dll', p.DLLS)
        self.assertTrue(set(p.COMPILER_HASHES) <= set(p.DLLS))

    def test_provenance_required_and_public(self):
        commit = 'a' * 40
        info = {'source_commit': commit, 'executable_sha256': 'b' * 64,
                'configuration': {'build_type': 'Release', 'cuda': True, 'tokenizer': True,
                                  'product_cli': True, 'gpu_architectures': ['86'], 'cuda_runtime_linkage': 'Shared'},
                'toolchain': {name: 'test-version' for name in
                              ('msvc', 'msvc_toolset', 'cmake', 'ninja', 'rust', 'cuda_toolkit', 'nvcc', 'cudnn')}}
        p.validate_build_info(info, commit)
        for name in info['toolchain']:
            incomplete = copy.deepcopy(info)
            del incomplete['toolchain'][name]
            with self.assertRaisesRegex(ValueError, 'missing toolchain'):
                p.validate_build_info(incomplete, commit)
        for private in ('C:\\Users\\someone\\build', '/home/someone/build', '\\\\server\\share',
                        'https://' + 'user:password' + '@example.invalid/file', 'ghp_' + 'a' * 36):
            with self.assertRaises(ValueError):
                p.public_metadata({'origin': private})
        p.public_metadata({'origin': 'https://example.invalid/source.tar.gz; FFmpeg / x264'})

    def test_canonical_git_resources_ignore_checkout_bytes(self):
        def git(*args, data=None):
            return subprocess.check_output(['git', '-C', str(self.root), *args], input=data)
        git('init', '-q')
        canonical = b'exact LF bytes\n'
        blob = git('hash-object', '-w', '--stdin', data=canonical).strip()
        def tree(entries):
            return git('mktree', data=b''.join(mode + b' ' + kind + b' ' + oid + b'\t' + name + b'\n'
                                             for mode, kind, oid, name in entries)).strip()
        specs = tree([(b'100644', b'blob', blob, f'{i:03d}.json'.encode()) for i in range(83)])
        native = tree([(b'040000', b'tree', specs, b'specs')])
        root_tree = tree([(b'100644', b'blob', blob, n.encode()) for n in p.ROOT_RESOURCES]
                         + [(b'040000', b'tree', native, b'native')])
        (self.root / 'VERSION').write_bytes(b'uncommitted different bytes\r\n')
        resources = p.canonical_resources(self.root, root_tree.decode())
        self.assertEqual(len(resources), 87)
        self.assertTrue(all(data == canonical for data in resources.values()))

    def test_zip_determinism_and_no_overwrite(self):
        stage = self.root / 'package'
        stage.mkdir()
        (stage / '目录.txt').write_bytes(b'license\n')
        a, b = self.root / 'a.zip', self.root / 'b.zip'
        p.write_zip(stage, a)
        p.write_zip(stage, b)
        self.assertEqual(p.digest(a), p.digest(b))
        with zipfile.ZipFile(a) as archive:
            self.assertEqual(archive.read('package/目录.txt'), b'license\n')
        with self.assertRaises(ValueError):
            p.write_zip(stage, a)


class InputRejectionTests(unittest.TestCase):
    """The five input rejection cases run in CI without proprietary DLLs."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = bytes(fixture())
        checksum = hashlib.sha256(data).hexdigest()
        license_path = self.root / 'licenses/test.txt'
        license_path.parent.mkdir()
        license_path.write_bytes(b'test fixture')
        self.manifest = {'schema_version': 1, 'runtime': [], 'resources': [
            {'path': 'licenses/test.txt', 'sha256': p.digest(license_path), 'origin': 'unit test'}]}
        for name in (*p.DLLS, 'vrhino-ffmpeg.exe'):
            (self.root / name).write_bytes(data)
            self.manifest['runtime'].append({'path': name, 'sha256': checksum, 'version': 'test',
                                            'origin': 'unit test', 'license': 'test fixture',
                                            'license_files': ['licenses/test.txt']})
        (self.root / 'dependency-manifest.json').write_text('{}')

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_runtime(self):
        for name in p.DLLS:
            incomplete = copy.deepcopy(self.manifest)
            incomplete['runtime'] = [x for x in incomplete['runtime'] if x['path'] != name]
            with self.assertRaisesRegex(ValueError, 'exactly 39'):
                p.dependency_paths(self.root, incomplete)
        (self.root / 'archive.dll').unlink()
        with self.assertRaisesRegex(ValueError, 'missing file'):
            p.dependency_paths(self.root, self.manifest)

    def test_hash_and_nvrtc_substitution(self):
        altered = copy.deepcopy(self.manifest)
        altered['runtime'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            p.dependency_paths(self.root, altered)
        # Every manifest hash matches its fixture bytes; the independent NVRTC
        # pins must still reject substitution, even with a changed manifest.
        with self.assertRaisesRegex(ValueError, 'NVRTC identity differs'):
            p.dependency_paths(self.root, self.manifest)

    def test_duplicate_resource(self):
        duplicate = dict(self.manifest['resources'][0])
        duplicate['path'] = duplicate['path'].upper()
        self.manifest['resources'].append(duplicate)
        with self.assertRaisesRegex(ValueError, 'case-colliding/duplicate'):
            p.dependency_paths(self.root, self.manifest)

    def test_unsafe_resource(self):
        for name in ('../escape.txt', 'sources/weights.pt', 'sources/video.wav'):
            altered = copy.deepcopy(self.manifest)
            altered['resources'][0]['path'] = name
            if not name.startswith('../'):
                dest = self.root / name
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(b'forbidden fixture')
            with self.assertRaisesRegex(ValueError, 'unsafe relative path|forbidden resource payload'):
                p.dependency_paths(self.root, altered)

    def test_unexpected_input(self):
        (self.root / 'nvJitLink_120_0.dll').write_bytes(b'unapproved')
        with self.assertRaisesRegex(ValueError, 'unexpected dependency-root file'):
            p.dependency_paths(self.root, self.manifest)


if __name__ == '__main__':
    unittest.main()
