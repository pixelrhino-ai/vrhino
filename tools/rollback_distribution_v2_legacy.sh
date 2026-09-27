#!/bin/sh
# Atomically route managed Distribution v2 launchers to the preserved v0.9.1 installation.
set -eu
prefix=${VRHINO_V2_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/vrhino-v2}
bin_dir=${VRHINO_V2_BIN_DIR:-$HOME/.local/bin}
while [ "$#" -gt 0 ]; do
    case $1 in
        --prefix) [ "$#" -ge 2 ] || exit 2; prefix=$2; shift 2;;
        --bin-dir) [ "$#" -ge 2 ] || exit 2; bin_dir=$2; shift 2;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2;;
    esac
done
case "$prefix" in /*) ;; *) echo 'ERROR: prefix must be absolute' >&2; exit 2;; esac
case "$bin_dir" in /*) ;; *) echo 'ERROR: bin-dir must be absolute' >&2; exit 2;; esac
[ -f "$prefix/.vrhino-v2-store" ] || { echo 'ERROR: unmanaged store' >&2; exit 1; }
[ -L "$prefix/current" ] || { echo 'ERROR: active pointer missing' >&2; exit 1; }
for name in vrhino vrhino-wan-family-convert; do
    [ -f "$bin_dir/$name" ] &&
        grep -Fqx '# VRhino distribution v2 launcher' "$bin_dir/$name" ||
        { echo "ERROR: unmanaged command: $name" >&2; exit 1; }
    [ -f "$prefix/legacy/v0.9.1-alpha/bin/$name" ] &&
        grep -Fqx '# VRhino installer launcher v1' "$prefix/legacy/v0.9.1-alpha/bin/$name" ||
        { echo "ERROR: legacy rollback target missing: $name" >&2; exit 1; }
done
"$prefix/legacy/v0.9.1-alpha/bin/vrhino" --version | grep -Fq 'v0.9.1-alpha' ||
    { echo 'ERROR: legacy installation cannot run' >&2; exit 1; }
lock=$prefix.install-lock
mkdir -- "$lock" 2>/dev/null || { echo 'ERROR: another install is running' >&2; exit 1; }
work=
cleanup() {
    code=$?
    trap - EXIT
    [ -z "$work" ] || rm -rf -- "$work"
    rmdir -- "$lock" 2>/dev/null || :
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
work=$(mktemp -d "$prefix/.rollback.XXXXXXXX")
ln -s 'legacy/v0.9.1-alpha' "$work/current"
mv -Tf -- "$work/current" "$prefix/current"
echo 'Activated preserved v0.9.1-alpha monolithic installation'
