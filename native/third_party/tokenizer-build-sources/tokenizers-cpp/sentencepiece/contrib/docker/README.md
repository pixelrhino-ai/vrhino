# Dockerfile

This directory contains a convenient Dockerfile to build and run sentencepiece C++ command line tools in a container.
**IMPORTANT**: this Dockerfile is community-maintained on a *best effort* basis only.
The sentencepiece core team does not take responsibility for ongoing maintenance, security fixes, or guarantees of uptime/compatibility.
Use at your own risk.

## Quickstart

Build locally (under the repository root directory):

```bash
docker build -t sentencepiece -f contrib/docker/Dockerfile .
```

[Run](https://docs.docker.com/reference/cli/docker/container/run/) the command line tools:

```bash
docker run --rm sentencepiece spm_encode --help
```
