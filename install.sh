#!/bin/sh
# Versioned Distribution v2 bootstrap. Keep execution inside the function so a
# truncated curl pipe response cannot start an installation.
vrhino_install_main() (
    set -eu
    release=v0.9.2-alpha
    base=https://github.com/pixelrhino-ai/vrhino/releases/download/$release
    catalog_name=vrhino-distribution-v2-catalog-v0.9.2-alpha.txt
    catalog_sha=d3890bd1b0d2d95f7be3500eb861abd84d2be4db4b77d70dfc600966d3a0fa80
    installer_name=install_distribution_v2.sh
    installer_sha=b1780b4e1bc652b5a954a713497abf65b7f95665e814d48ca78e8e8c95b436e7
    bundle=
    prefix=
    bin_dir=
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
                printf 'Online release: %s\n' "$release"
                exit 0;;
            *) fail "unknown option: $1";;
        esac
    done
    command -v sha256sum >/dev/null 2>&1 || fail 'sha256sum is required'
    if [ -n "$bundle" ]; then
        installer=$bundle/$installer_name
        catalog=$bundle/catalog.txt
        [ -f "$installer" ] || fail "offline installer missing: $installer"
        [ -f "$catalog" ] || fail "offline catalog missing: $catalog"
        actual=$(sha256sum < "$catalog"); actual=${actual%% *}
        [ "$actual" = "$catalog_sha" ] || fail 'offline catalog SHA256 mismatch'
    else
        command -v curl >/dev/null 2>&1 || fail 'curl is required for online install'
        work=$(mktemp -d "${TMPDIR:-/tmp}/vrhino-v2-bootstrap.XXXXXXXX")
        installer=$work/$installer_name
        curl --silent --show-error --fail --location --proto '=https' \
            --proto-redir '=https' --tlsv1.2 --retry 3 --retry-all-errors \
            --retry-delay 1 --connect-timeout 30 --output "$installer" \
            "$base/$installer_name" || fail 'installer download failed'
    fi
    actual=$(sha256sum < "$installer"); actual=${actual%% *}
    [ "$actual" = "$installer_sha" ] || fail 'installer SHA256 mismatch'
    set --
    if [ -n "$prefix" ]; then set -- "$@" --prefix "$prefix"; fi
    if [ -n "$bin_dir" ]; then set -- "$@" --bin-dir "$bin_dir"; fi
    if [ -n "$bundle" ]; then
        set -- "$@" --bundle-dir "$bundle" --catalog-sha256 "$catalog_sha"
    else
        set -- "$@" --catalog-url "$base/$catalog_name" --catalog-sha256 "$catalog_sha"
    fi
    sh "$installer" "$@" --expected-catalog-version "$release"
    printf 'If needed in this terminal: export PATH="%s:$PATH"\n' "${bin_dir:-$HOME/.local/bin}"
)
vrhino_install_main "$@"
