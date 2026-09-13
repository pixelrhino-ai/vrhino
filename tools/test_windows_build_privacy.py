"""Native MSVC build-policy regression; requires cl, CMake and Ninja on PATH."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('packager', Path(__file__).with_name('package_windows_build.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


@unittest.skipUnless(os.name == 'nt', 'native MSVC test')
class BuildPrivacyTests(unittest.TestCase):
    def test_real_project_header_generated_and_wide_locations(self):
        with tempfile.TemporaryDirectory(prefix='vrhino privacy ') as directory:
            root = Path(directory)
            source, build = root / 'source', root / 'build'
            source.mkdir()
            (source / 'third party').mkdir()
            (source / 'third party/header.h').write_text(
                '#include <stdexcept>\n#include <string>\n'
                'inline void diagnostic(){throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": diagnostic retained");}\n', encoding='utf-8')
            (source / 'probe.cpp').write_text(
                '#include <stdio.h>\n#include "header.h"\n#include "generated.h"\n'
                '#define W2(x) L##x\n#define W(x) W2(x)\n'
                'int main(){puts(__FILE__);printf("%ls\\n", W(__FILE__));puts(generated_file());'
                'try{diagnostic();}catch(const std::exception& e){puts(e.what());}}\n', encoding='utf-8')
            module = Path(__file__).resolve().parents[1] / 'native/cmake/windows_build_privacy.cmake'
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.22)\nproject(privacy CXX)\n'
                f'include("{module.as_posix()}")\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/generated.h" "inline const char* generated_file(){return __FILE__;}\\n")\n'
                'add_executable(probe probe.cpp)\n'
                'target_include_directories(probe PRIVATE "${CMAKE_SOURCE_DIR}/third party" "${CMAKE_BINARY_DIR}")\n'
                'target_compile_options(probe PRIVATE /FC /Zi)\n'
                'target_link_options(probe PRIVATE /DEBUG)\n', encoding='utf-8')
            def run(command):
                result = subprocess.run(command, capture_output=True)
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors='replace'))
                return result.stdout.decode(errors='replace').replace('\\', '/')
            run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release'])
            run(['cmake', '--build', str(build)])
            output = run([str(build / 'probe.exe')])
            self.assertEqual(output.splitlines(), ['vrhino/source/probe.cpp', 'vrhino/source/probe.cpp',
                                                  'build/generated.h', 'vrhino/source/third party/header.h:3: diagnostic retained'])
            self.assertEqual(p.binary_private_paths(build / 'probe.exe', [root]), [])
            # Source names in debug records are remapped; PDBs can additionally
            # retain compiler command lines and are explicitly not distributed.
            pdb = (build / 'probe.pdb').read_bytes()
            self.assertIn(b'vrhino', pdb)
            self.assertNotIn(str(source / 'probe.cpp').encode(), pdb)


if __name__ == '__main__':
    unittest.main()
