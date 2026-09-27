#!/bin/sh
# Show installed core versions that still reference an exact NVIDIA runtime ID.
set -eu
prefix=${1:-${VRHINO_V2_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/vrhino-v2}}
id=${2:-}
[ -n "$id" ] || { echo 'Usage: sh distribution_v2_references.sh PREFIX RUNTIME_ID' >&2; exit 2; }
case "$id" in *[!a-zA-Z0-9._-]*|'') exit 2;; esac
for requirement in "$prefix"/runtime/*/requires-nvidia.txt; do
    [ -f "$requirement" ] || continue
    if grep -Fqx "id=$id" "$requirement"; then
        version=${requirement#"$prefix"/runtime/}
        printf '%s\n' "${version%/requires-nvidia.txt}"
    fi
done
