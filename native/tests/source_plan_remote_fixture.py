"""Exercise Native declared-plan acquisition over a real loopback HTTP transport."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


class Handler(SimpleHTTPRequestHandler):
    requests = []

    def do_GET(self):
        self.requests.append(self.path)
        super().do_GET()

    def log_message(self, *_args):
        pass


def main():
    with tempfile.TemporaryDirectory(prefix='vrhino-source-remote-') as directory:
        relative = '/owner/model/resolve/' + 'a' * 40 + '/test.bin'
        payload = Path(directory) / relative.lstrip('/')
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b'test')
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=directory))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            subprocess.run([sys.argv[1], f'http://127.0.0.1:{server.server_port}'],
                           check=True, timeout=120)
            assert relative in Handler.requests, 'native acquisition made no source request'
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print('declared source-plan remote transport fixture: PASS')


if __name__ == '__main__':
    main()
