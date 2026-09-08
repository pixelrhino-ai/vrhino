# cpp-httplib upstream record

- Project: cpp-httplib
- Upstream: https://github.com/yhirose/cpp-httplib
- Stable release: `v0.54.1`
- Upstream commit: `9d6a7ee2c1aaeb1fd9ae15d14f06f487149d147f`
- Source archive: https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.54.1.tar.gz
- Archive SHA-256: `7310f5312e1423830d649b38ed028e9db86303a979ccbfdbd1c4b1574f422dfb`
- License: MIT

Vendored files are limited to the unmodified single-header implementation and
upstream license:

- `httplib.h`: 784,467 bytes, SHA-256
  `5933c14b2d0f45212925ed18ca579841f5fce717f431fc20cec712423e905b10`
- `LICENSE`: 1,075 bytes, SHA-256
  `4b45cbe16d7b71b89ae6127e26e0d90a029198ca5e958ad8e3d0b8bbed364d8b`

VRhino compiles the header in plain HTTP mode. TLS, zlib, Brotli, and other
optional compression integrations are not enabled. VRhino applies smaller
request, header, body, backlog, worker, queue, and timeout bounds in its own
Native API target and server configuration.
