#!/bin/sh
# Versioned Distribution v2 bootstrap. Its pins are frozen by the release tag.
# Function wrapping prevents a truncated curl pipe from starting installation.
vrhino_install_main() (
    set -eu
    release=v0.9.3-alpha
    core_name=vrhino-core-linux-x86_64-v0.9.3-alpha.tar.gz
    core_sha=3f3ffc3fc1676c1616dac48749643136509ce299aec85e1d94479ccd9f26f4b0
    core_root=vrhino-core-v0.9.3-alpha
    core_version=v0.9.3-alpha
    runtime_name=vrhino-nvidia-runtime-linux-x86_64-cuda12.8-cudnn9.8-r1.tar.gz
    runtime_sha=c3488da9a4d346ff791cb14ba012018e50b7876ab7dd4dbecc7ba53e088dbe83
    runtime_root=vrhino-nvidia-runtime-nvidia-linux-x86_64-cuda12.8.90-cublas12.8.4.1-cudnn9.8.0.87-nvrtc12.8.93-nvjitlink12.8.93-r1
    runtime_id=nvidia-linux-x86_64-cuda12.8.90-cublas12.8.4.1-cudnn9.8.0.87-nvrtc12.8.93-nvjitlink12.8.93-r1
    helper_name=install_distribution_v2.sh
    helper_sha=f6e5a44ab51f9ad7d4a331f5ec15773d33cfc9af8531c7fc186e67198ed7f042
    release_base=https://github.com/pixelrhino-ai/vrhino/releases/download/$release
    helper_url=https://raw.githubusercontent.com/pixelrhino-ai/vrhino/$release/tools/$helper_name
    prefix=
    bin_dir=
    bundle=
    work=
    fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
    cleanup() {
        code=$?
        trap - EXIT
        [ -z "$work" ] || rm -rf -- "$work"
        exit "$code"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM HUP
    while [ "$#" -gt 0 ]; do
        case $1 in
            --prefix|--bin-dir|--bundle-dir)
                [ "$#" -ge 2 ] || fail "missing value for $1"
                case $1 in
                    --prefix) prefix=$2;;
                    --bin-dir) bin_dir=$2;;
                    --bundle-dir) bundle=$2;;
                esac
                shift 2;;
            --help|-h)
                printf 'Usage: sh install.sh [--prefix ABSOLUTE_DIR] [--bin-dir ABSOLUTE_DIR]\n'
                printf 'Offline: sh install.sh --bundle-dir DIR [--prefix DIR] [--bin-dir DIR]\n'
                printf 'Release: %s; binary build: v0.9.1-alpha\n' "$release"
                exit 0;;
            *) fail "unknown option: $1";;
        esac
    done
    command -v sha256sum >/dev/null 2>&1 || fail 'sha256sum is required'
    work=$(mktemp -d "${TMPDIR:-/tmp}/vrhino-v2-bootstrap.XXXXXXXX")
    catalog=$work/catalog.txt
    cat > "$catalog" <<EOF
schema=vrhino-distribution-v2-catalog-v1
catalog_version=$release
platform=linux-x86_64
runtime_api=1
core_name=$core_name
core_sha256=$core_sha
core_root=$core_root
core_version=$core_version
runtime_name=$runtime_name
runtime_sha256=$runtime_sha
runtime_root=$runtime_root
runtime_id=$runtime_id
core_url=$release_base/$core_name
runtime_url=$release_base/$runtime_name
EOF
    if [ -n "$bundle" ]; then
        helper=$bundle/$helper_name
        [ -f "$helper" ] || fail "offline helper missing: $helper"
    else
        command -v curl >/dev/null 2>&1 || fail 'curl is required for online install'
        helper=$work/$helper_name
        curl --silent --show-error --fail --location --proto '=https' \
            --proto-redir '=https' --tlsv1.2 --retry 3 --retry-all-errors \
            --retry-delay 1 --connect-timeout 30 --output "$helper" \
            "$helper_url" || fail 'versioned installer helper download failed'
    fi
    actual=$(sha256sum < "$helper"); actual=${actual%% *}
    [ "$actual" = "$helper_sha" ] || fail 'installer helper SHA256 mismatch'
    set -- --catalog "$catalog" --expected-catalog-version "$release"
    if [ -n "$prefix" ]; then set -- "$@" --prefix "$prefix"; fi
    if [ -n "$bin_dir" ]; then set -- "$@" --bin-dir "$bin_dir"; fi
    if [ -n "$bundle" ]; then set -- "$@" --bundle-dir "$bundle"; fi
    sh "$helper" "$@"
    printf 'Installed distribution %s (binary build v0.9.1-alpha).\n' "$release"
    printf 'If needed in this terminal: export PATH="%s:$PATH"\n' "${bin_dir:-$HOME/.local/bin}"
)
vrhino_install_main "$@"
