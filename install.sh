#!/bin/sh
# Install the published Linux alpha. No model download, sudo, or inference.
# Keep all execution inside main so a truncated curl response cannot partly install.
vrhino_install_main() (
    set -eu
    version=v0.9.0-alpha
    archive_name=vrhino-linux-x86_64-cuda-v0.9.0-alpha-candidate.tar.gz
    archive_sha=e531f218fecf8316145ee336b009da9826c79dbc7b704736a8d455449b96c0fd
    archive_root=vrhino-v0.9.0-alpha
    url="https://github.com/pixelrhino-ai/vrhino/releases/download/$version/$archive_name"
    prefix=${VRHINO_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/$archive_root}
    bin_dir=${VRHINO_BIN_DIR:-$HOME/.local/bin}
    local_archive=
    modify_path=yes
    login_shell=${SHELL:-}
    work=
    lock=
    fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
    say() { printf '%s\n' "$*"; }
    quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
    cleanup() {
        code=$?
        trap - EXIT
        [ -z "$work" ] || rm -rf -- "$work"
        [ -z "$lock" ] || rmdir -- "$lock" 2>/dev/null || :
        [ "$code" -eq 0 ] || printf 'Installation stopped; no existing installation was deleted.\n' >&2
        exit "$code"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM HUP
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --prefix|--bin-dir|--archive)
                [ "$#" -ge 2 ] || fail "Missing value for $1"
                case "$1" in --prefix) prefix=$2;; --bin-dir) bin_dir=$2;; --archive) local_archive=$2;; esac
                shift 2;;
            --no-modify-path) modify_path=no; shift;;
            --help|-h)
                say 'Usage: sh install.sh [--prefix ABSOLUTE_DIR] [--bin-dir ABSOLUTE_DIR] [--archive FILE] [--no-modify-path]'
                say 'Default: ~/.local/share/vrhino-v0.9.0-alpha and ~/.local/bin. Installs Linux x86_64 only.'
                exit 0;;
            *) fail "Unknown option: $1";;
        esac
    done
    say '[1/5] Checking platform and installation paths'
    [ "$(uname -s)" = Linux ] || fail 'This installer supports Linux only. Windows: see docs/install.md.'
    [ "$(uname -m)" = x86_64 ] || fail 'The published package requires x86_64.'
    for command in curl tar sha256sum getconf awk sed grep mktemp mkdir mv chmod dirname; do
        command -v "$command" >/dev/null 2>&1 || fail "Required command missing: $command"
    done
    glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null) || fail 'glibc 2.35 or newer is required.'
    printf '%s\n' "$glibc" | awk '$1=="glibc" {split($2,v,"."); if(v[1]>2 || (v[1]==2 && v[2]>=35)) ok=1} END {exit !ok}' || fail "glibc 2.35 or newer is required; found $glibc"
    for path in "$prefix" "$bin_dir"; do
        case "$path" in /*) ;; *) fail 'Installation directories must be absolute paths.';; esac
        case "$path" in /|*'
'*) fail 'Invalid installation directory.';; esac
    done
    [ ! -L "$prefix" ] || fail "Refusing symlink installation directory: $prefix"
    mkdir -p -- "$(dirname -- "$prefix")" "$bin_dir"
    # Canonicalize parent directories without following a pre-existing prefix link.
    parent=$(CDPATH= cd -- "$(dirname -- "$prefix")" && pwd -P)
    leaf=${prefix##*/}; [ -n "$leaf" ] && [ "$leaf" != . ] && [ "$leaf" != .. ] || fail 'Invalid prefix leaf.'
    prefix=$parent/$leaf
    bin_dir=$(CDPATH= cd -- "$bin_dir" && pwd -P)
    lock_path=$prefix.install-lock
    mkdir -- "$lock_path" 2>/dev/null || fail "Another installation may be running. Lock: $lock_path"
    lock=$lock_path
    for name in vrhino vrhino-wan-family-convert; do
        target=$bin_dir/$name
        if [ -e "$target" ] || [ -L "$target" ]; then
            [ ! -L "$target" ] && [ -f "$target" ] && grep -Fqx '# VRhino installer launcher v1' "$target" || fail "Refusing to overwrite existing command: $target"
        fi
    done
    if [ -e "$prefix" ]; then
        [ -d "$prefix" ] && [ -f "$prefix/.vrhino-install-identity" ] || fail "Directory already exists and is not managed by this installer: $prefix"
        [ "$(cat "$prefix/.vrhino-install-identity")" = "$archive_sha" ] || fail 'Existing installation has a different identity; use a new --prefix.'
        say '[2/5] Existing installation found; checking package checksums'
        (cd "$prefix" && sha256sum --quiet -c SHA256SUMS) || fail 'Existing package is damaged; use a new --prefix.'
    else
        work=$(mktemp -d "$parent/.vrhino-install.XXXXXXXX")
        if [ -n "$local_archive" ]; then
            [ -f "$local_archive" ] || fail "Archive not found: $local_archive"
            archive=$local_archive
            say '[2/5] Checking local archive'
        else
            archive=$work/$archive_name
            say '[2/5] Downloading Linux CUDA package (about 1.45 GB)'
            curl --fail --location --proto '=https' --proto-redir '=https' --tlsv1.2 \
                --retry 3 --connect-timeout 30 --progress-bar --output "$archive" "$url" || fail 'Download failed; rerun to retry.'
        fi
        actual=$(sha256sum < "$archive"); actual=${actual%% *}
        [ "$actual" = "$archive_sha" ] || fail 'Archive SHA256 mismatch; nothing was extracted.'
        say '[3/5] SHA256 verified; extracting and checking package'
        tar -xzf "$archive" -C "$work" --no-same-owner
        unpacked=$work/$archive_root
        [ -d "$unpacked" ] && [ ! -L "$unpacked" ] || fail 'Unexpected archive layout.'
        (cd "$unpacked" && sha256sum --quiet -c SHA256SUMS) || fail 'Extracted package checksum failure.'
        for name in vrhino vrhino-wan-family-convert; do
            [ -x "$unpacked/bin/$name" ] || fail "Missing package launcher: $name"
        done
        detected=$("$unpacked/bin/vrhino" --version) || fail 'Packaged executable cannot start on this host.'
        printf '%s\n' "$detected" | grep -Fqx "VRhino $version" || fail 'Unexpected executable version.'
        printf '%s\n' "$archive_sha" > "$unpacked/.vrhino-install-identity"
        mv -T -- "$unpacked" "$prefix"
    fi
    say '[4/5] Installing commands'
    # A symlink to the shipped shell wrapper would break its relative library paths.
    for name in vrhino vrhino-wan-family-convert; do
        launcher=$(mktemp "$bin_dir/.vrhino-launcher.XXXXXXXX")
        {
            printf '#!/bin/sh\n# VRhino installer launcher v1\nexec '
            quote "$prefix/bin/$name"
            printf ' "$@"\n'
        } > "$launcher"
        chmod 755 "$launcher"
        mv -T -- "$launcher" "$bin_dir/$name"
    done
    "$bin_dir/vrhino" --version
    say '[5/5] Configuring command search path'
    path_line="export PATH=$(quote "$bin_dir"):\"\$PATH\""
    if [ "$modify_path" = yes ]; then
        for profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc"; do
            case "$profile" in
                */.bashrc) [ -f "$profile" ] || [ "${login_shell##*/}" = bash ] || continue;;
                */.zshrc) [ -f "$profile" ] || [ "${login_shell##*/}" = zsh ] || continue;;
            esac
            if ! grep -Fqx "$path_line" "$profile" 2>/dev/null; then
                printf '\n# VRhino command path\n%s\n' "$path_line" >> "$profile"
            fi
        done
    fi
    say "SUCCESS: VRhino $version installed at $prefix"
    case ":$PATH:" in
        *":$bin_dir:"*) say 'Next: vrhino doctor';;
        *) say 'For this terminal, run:'; say "  $path_line"; say 'Then: vrhino doctor';;
    esac
    [ "$modify_path" = no ] || say 'New Bash/Zsh login or interactive terminals will use the configured PATH.'
    say 'Wan2.2 setup: https://github.com/pixelrhino-ai/vrhino/blob/main/docs/models/wan2.2-quickstart.md'
)
vrhino_install_main "$@"
