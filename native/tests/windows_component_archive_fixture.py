"""Small tar.gz fixtures and loopback registry; never extracts or launches archive contents."""
import gzip
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading


def main():
    executable = Path(sys.argv[1]).resolve()
    root = executable.parent / "archive fixtures Unicode 测试"
    root.mkdir(exist_ok=True)
    payload = b"component fixture: not an executable\n"
    manifest = {
        "schema_version": 1,
        "identity": {"namespace": "vrhino", "name": "media-windows-x86_64", "version": "1.0.0", "publisher": "test"},
        "contract": {"name": "vrhino.media.rgb24-h264-mp4.cli", "major": 1, "minor": 0},
        "platform": {"os": "windows", "architecture": "x86_64"},
        "entrypoint": "bin/vrhino-ffmpeg.exe",
        "artifacts": [{"path": "bin/vrhino-ffmpeg.exe", "size": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest(), "executable": True}],
    }
    cases = []

    def make(name, extras=(), document=None, valid=False):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
            entries = [("vrhino-media/vrhino-component.json", json.dumps(document or manifest, ensure_ascii=False).encode(), tarfile.REGTYPE, ""),
                       ("vrhino-media/bin/vrhino-ffmpeg.exe", payload, tarfile.REGTYPE, "")]
            for path, data, kind, link in entries + list(extras):
                item = tarfile.TarInfo(path)
                item.type = kind
                item.mode = 0o755 if path.endswith(".exe") else 0o644
                item.linkname = link
                item.size = len(data)
                if name == "sparse" and path.endswith("/sparse.bin"):
                    item.pax_headers = {"GNU.sparse.size": "1048576", "GNU.sparse.map": "0,3,1048573,3"}
                archive.addfile(item, io.BytesIO(data))
        (root / (name + ".tar.gz")).write_bytes(gzip.compress(output.getvalue(), mtime=0))
        cases.append((name, valid))

    make("valid", valid=True)
    make("Unicode 档案", [("vrhino-media/目录/媒体 文件.txt", b"unicode", tarfile.REGTYPE, "")], valid=True)
    make("long", [("vrhino-media/" + "/".join(["segment" * 12] * 4) + "/file.txt", b"long", tarfile.REGTYPE, "")], valid=True)
    make("hardlink", [("vrhino-media/alias", b"", tarfile.LNKTYPE, "vrhino-media/bin/vrhino-ffmpeg.exe")], valid=True)
    make("forward-link", [("vrhino-media/alias", b"", tarfile.LNKTYPE, "vrhino-media/later"),
                          ("vrhino-media/later", b"forward", tarfile.REGTYPE, "")], valid=True)
    make("empty", [("vrhino-media/empty", b"", tarfile.REGTYPE, ""),
                   ("vrhino-media/empty-dir/", b"", tarfile.DIRTYPE, "")], valid=True)
    make("sparse", [("vrhino-media/sparse.bin", b"abcxyz", tarfile.REGTYPE, "")], valid=True)
    unicode_manifest = json.loads(json.dumps(manifest))
    unicode_manifest["entrypoint"] = "工具/媒体 helper.exe"
    unicode_manifest["artifacts"][0]["path"] = unicode_manifest["entrypoint"]
    make("unicode-manifest", [("vrhino-media/工具/媒体 helper.exe", payload, tarfile.REGTYPE, "")],
         document=unicode_manifest, valid=True)
    paths = ["/escape", "C:/escape", "C:escape", "//server/share/escape", "../escape",
             "vrhino-media/../escape", "vrhino-media/./file", "vrhino-media//file",
             "vrhino-media/CON", "vrhino-media/nul.txt", "vrhino-media/COM1.exe", "vrhino-media/LPT9",
             "vrhino-media/COM¹.txt", "vrhino-media/CONIN$", "vrhino-media/file:stream",
             "vrhino-media/a?b", "vrhino-media/a*b", "vrhino-media/a<b", "vrhino-media/a>b",
             'vrhino-media/a"b', "vrhino-media/a|b", "vrhino-media/a\\..\\b",
             "vrhino-media/trailing.", "vrhino-media/trailing ", "vrhino-media/control\x01",
             "vrhino-media/bin/vrhino-ffmpeg.exe", "vrhino-media/BIN/other", "outside/file", "vrhino-media/CON .txt"]
    for index, path in enumerate(paths):
        make("unsafe-" + str(index), [(path, b"bad", tarfile.REGTYPE, "")])
    make("symlink", [("vrhino-media/link", b"", tarfile.SYMTYPE, "../../outside")])
    make("link-escape", [("vrhino-media/link", b"", tarfile.LNKTYPE, "../outside")])
    make("link-cycle", [("vrhino-media/link", b"", tarfile.LNKTYPE, "vrhino-media/link")])
    make("fifo", [("vrhino-media/fifo", b"", tarfile.FIFOTYPE, "")])
    make("type-collision", [("vrhino-media/bin", b"bad", tarfile.REGTYPE, "")])
    for name, edit in [
        ("bad-hash", lambda m: m["artifacts"][0].update(sha256="0" * 64)),
        ("bad-size", lambda m: m["artifacts"][0].update(size=(1 << 32) + 7)),
        ("bad-identity", lambda m: m["identity"].update(name="../escape")),
        ("reserved-identity", lambda m: m["identity"].update(name="nul")),
        ("bad-platform", lambda m: m["platform"].update(os="linux")),
        ("unsafe-manifest", lambda m: m["artifacts"][0].update(path="bin/vrhino-ffmpeg.exe:ads")),
        ("duplicate-manifest", lambda m: m["artifacts"].append(dict(m["artifacts"][0], path="BIN/vrhino-ffmpeg.exe"))),
    ]:
        changed = json.loads(json.dumps(manifest))
        edit(changed)
        make(name, document=changed)
    good = (root / "valid.tar.gz").read_bytes()
    expanded = gzip.decompress(good)
    (root / "gzip-members.tar.gz").write_bytes(gzip.compress(expanded[:512]) + gzip.compress(expanded[512:]))
    cases.append(("gzip-members", True))
    for name, data in [("truncated", good[:-5]), ("crc", good[:-8] + bytes([good[-8] ^ 1]) + good[-7:]),
                       ("trailing-data", good + b"untrusted"), ("raw-tar", gzip.decompress(good)),
                       ("corrupt", b"not an archive")]:
        (root / (name + ".tar.gz")).write_bytes(data)
        cases.append((name, False))
    # A tiny fixture with a real 64-bit tar size declaration, no large download.
    header = tarfile.TarInfo("vrhino-media/large")
    header.size = (1 << 32) + 17
    (root / "large-truncated.tar.gz").write_bytes(gzip.compress(header.tobuf(tarfile.GNU_FORMAT) + bytes(1024)))
    cases.append(("large-truncated", False))
    (root / "cases.txt").write_text("\n".join(f"{int(ok)} {name}" for name, ok in cases), encoding="utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.endswith("/index.json"):
                data = json.dumps({"registry_schema_version": 1,
                                   "identity": "vrhino/media-windows-x86_64:1.0.0",
                                   "package": {"url": base + "/archive", "size": len(good),
                                               "sha256": hashlib.sha256(good).hexdigest()}}).encode()
            elif self.path == "/archive":
                data = good
            else:
                self.send_error(404)
                return
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            self.send_response(206 if "Range" in self.headers else 200)
            if "Range" in self.headers:
                self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            try:
                self.wfile.write(data[start:])
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = subprocess.run([str(executable), str(root), base], timeout=180,
                                creationflags=subprocess.CREATE_NO_WINDOW,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        print(result.stdout, end="", flush=True)
        if result.returncode:
            print(f"native exit code: {result.returncode:#x}", flush=True)
        return result.returncode
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
