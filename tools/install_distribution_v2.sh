#!/bin/sh
# Distribution v2 prototype installer. A catalog pins both immutable archives.
install_main() (
    set -eu
    prefix=${VRHINO_V2_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/vrhino-v2}
    bin_dir=${VRHINO_V2_BIN_DIR:-$HOME/.local/bin}
    bundle=
    catalog=
    catalog_url=
    catalog_sha=
    expected_catalog_version=
    ca_file=
    catalog_work=
    work=
    lock=
    lock_owned=0
    legacy_migration=0
    activation_committed=0
    legacy_current_created=0
    initial_current_created=0
    installed_commands=
    fail() { echo "ERROR: $*" >&2; exit 1; }
    quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
    render_launcher() {
        name=$1
        printf '#!/bin/sh\n# VRhino distribution v2 launcher\nexec '
        quote "$prefix/current/bin/$name"
        printf ' "$@"\n'
    }
    cleanup() {
        code=$?
        trap - EXIT
        if [ "$code" -ne 0 ] && [ "$activation_committed" -eq 0 ]; then
            if [ "$legacy_migration" -eq 1 ] && [ -d "$prefix/legacy/v0.9.1-alpha/bin" ]; then
                for name in vrhino vrhino-wan-family-convert; do
                    if [ -f "$prefix/legacy/v0.9.1-alpha/bin/$name" ]; then
                        cp -p -- "$prefix/legacy/v0.9.1-alpha/bin/$name" "$bin_dir/$name.restore-$$" &&
                            mv -Tf -- "$bin_dir/$name.restore-$$" "$bin_dir/$name" || :
                    fi
                done
                if [ "$legacy_current_created" -eq 1 ] &&
                   [ "$(readlink "$prefix/current" 2>/dev/null || :)" = 'legacy/v0.9.1-alpha' ]; then
                    rm -f -- "$prefix/current"
                fi
            elif [ "$initial_current_created" -eq 1 ]; then
                for name in $installed_commands; do rm -f -- "$bin_dir/$name"; done
                if [ "$(readlink "$prefix/current" 2>/dev/null || :)" = "runtime/$core_version" ]; then
                    rm -f -- "$prefix/current"
                fi
            fi
        fi
        [ -z "$work" ] || rm -rf -- "$work"
        [ -z "$catalog_work" ] || rm -rf -- "$catalog_work"
        if [ "$lock_owned" -eq 1 ]; then
            rm -f -- "$lock/pid"
            rmdir -- "$lock" 2>/dev/null || :
        fi
        exit "$code"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM HUP
    while [ "$#" -gt 0 ]; do
        case $1 in
            --prefix|--bin-dir|--bundle-dir|--catalog|--catalog-url|--catalog-sha256|--expected-catalog-version|--ca-file)
                [ "$#" -ge 2 ] || fail "missing value for $1"
                case $1 in
                    --prefix) prefix=$2;;
                    --bin-dir) bin_dir=$2;;
                    --bundle-dir) bundle=$2;;
                    --catalog) catalog=$2;;
                    --catalog-url) catalog_url=$2;;
                    --catalog-sha256) catalog_sha=$2;;
                    --expected-catalog-version) expected_catalog_version=$2;;
                    --ca-file) ca_file=$2;;
                esac
                shift 2;;
            --help|-h)
                echo 'Usage: sh install_distribution_v2.sh --bundle-dir DIR [--prefix DIR] [--bin-dir DIR]'
                echo 'Or: sh install_distribution_v2.sh --catalog FILE [--prefix DIR] [--bin-dir DIR]'
                echo 'Or: sh install_distribution_v2.sh --catalog-url HTTPS_URL --catalog-sha256 SHA256 [--expected-catalog-version VERSION] [--ca-file PEM]'
                exit 0;;
            *) fail "unknown option: $1";;
        esac
    done
    [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || fail 'requires Linux x86_64'
    for cmd in sha256sum tar gzip grep sed mktemp mkdir mv dirname readlink rm chmod cat ln cp cmp; do
        command -v "$cmd" >/dev/null 2>&1 || fail "missing tool: $cmd"
    done
    if [ -n "$ca_file" ]; then [ -f "$ca_file" ] || fail 'CA file missing'; fi
    curl_fetch() {
        url=$1; destination=$2
        case "$url" in https://*) ;; *) fail 'artifact and catalog URLs must use HTTPS';; esac
        command -v curl >/dev/null 2>&1 || fail 'curl missing for download'
        if [ -n "$ca_file" ]; then
            curl --silent --show-error --fail --location --proto '=https' --proto-redir '=https' --tlsv1.2 \
                --retry 3 --retry-all-errors --retry-delay 1 --connect-timeout 30 \
                --cacert "$ca_file" --output "$destination" "$url" || fail "download failed: $url"
        else
            curl --silent --show-error --fail --location --proto '=https' --proto-redir '=https' --tlsv1.2 \
                --retry 3 --retry-all-errors --retry-delay 1 --connect-timeout 30 \
                --output "$destination" "$url" || fail "download failed: $url"
        fi
    }
    if [ -n "$catalog_url" ]; then
        [ -z "$catalog" ] && [ -z "$bundle" ] || fail 'choose one catalog source'
        [ -n "$catalog_sha" ] || fail 'online catalog requires a trusted SHA256 pin'
        catalog_work=$(mktemp -d "${TMPDIR:-/tmp}/vrhino-v2-catalog.XXXXXXXX")
        catalog=$catalog_work/catalog.txt
        curl_fetch "$catalog_url" "$catalog"
    fi
    if [ -n "$bundle" ]; then
        [ -z "$catalog" ] || fail 'choose bundle or catalog'
        catalog=$bundle/catalog.txt
    fi
    [ -f "$catalog" ] || fail 'catalog missing'
    if [ -n "$catalog_sha" ]; then
        [ "${#catalog_sha}" -eq 64 ] || fail 'invalid catalog SHA length'
        case "$catalog_sha" in *[!a-f0-9]*) fail 'invalid catalog SHA';; esac
        actual=$(sha256sum < "$catalog"); actual=${actual%% *}
        [ "$actual" = "$catalog_sha" ] || fail 'catalog SHA256 mismatch'
    fi
    seen='|'
    while IFS= read -r entry || [ -n "$entry" ]; do
        case "$entry" in *=*) key=${entry%%=*};; *) fail 'malformed catalog entry';; esac
        case "$key" in
            schema|catalog_version|platform|runtime_api|source_archive_sha256|core_name|core_sha256|core_root|core_version|runtime_name|runtime_sha256|runtime_root|runtime_id|core_url|runtime_url) ;;
            *) fail "unknown catalog key: $key";;
        esac
        case "$seen" in *"|$key|"*) fail "duplicate catalog key: $key";; esac
        seen=$seen$key'|'
    done < "$catalog"
    get() { sed -n "s/^$1=//p" "$catalog" | sed -n '1p'; }
    [ "$(get schema)" = vrhino-distribution-v2-catalog-v1 ] || fail 'unsupported catalog schema'
    [ "$(get platform)" = linux-x86_64 ] || fail 'unsupported catalog platform'
    [ "$(get runtime_api)" = 1 ] || fail 'unsupported catalog runtime API'
    catalog_version=$(get catalog_version)
    case "$catalog_version" in ''|*[!a-zA-Z0-9._-]*) fail 'invalid catalog version';; esac
    if [ -n "$expected_catalog_version" ]; then
        [ "$catalog_version" = "$expected_catalog_version" ] || fail 'stale or wrong catalog version'
    fi
    core_name=$(get core_name); core_sha=$(get core_sha256); core_root=$(get core_root)
    core_version=$(get core_version); runtime_name=$(get runtime_name)
    runtime_sha=$(get runtime_sha256); runtime_root=$(get runtime_root)
    runtime_id=$(get runtime_id); core_url=$(get core_url); runtime_url=$(get runtime_url)
    for value in "$core_name" "$core_root" "$core_version" "$runtime_name" "$runtime_root" "$runtime_id"; do
        case "$value" in ''|*[!a-zA-Z0-9._-]*) fail 'invalid catalog identity';; esac
    done
    for value in "$core_sha" "$runtime_sha"; do
        [ "${#value}" -eq 64 ] || fail 'invalid archive SHA length'
        case "$value" in *[!a-f0-9]*) fail 'invalid archive SHA';; esac
    done
    for value in "$prefix" "$bin_dir"; do
        case "$value" in /*) ;; *) fail 'installation paths must be absolute';; esac
        case "$value" in /|*'
'*) fail 'invalid installation path';; esac
    done
    [ ! -L "$prefix" ] || fail 'prefix is a symlink'
    parent=$(dirname -- "$prefix")
    mkdir -p -- "$parent" "$bin_dir"
    parent=$(CDPATH= cd -- "$parent" && pwd -P)
    prefix=$parent/${prefix##*/}
    bin_dir=$(CDPATH= cd -- "$bin_dir" && pwd -P)
    if [ -e "$prefix" ]; then
        [ -f "$prefix/.vrhino-v2-store" ] || fail 'unmanaged prefix'
    fi
    lock=$prefix.install-lock
    if ! mkdir -- "$lock" 2>/dev/null; then
        old_pid=$(cat "$lock/pid" 2>/dev/null || :)
        case "$old_pid" in ''|*[!0-9]*) fail 'install lock has no valid owner';; esac
        kill -0 "$old_pid" 2>/dev/null && fail 'another install is running'
        rm -f -- "$lock/pid"
        rmdir -- "$lock" 2>/dev/null || fail 'stale install lock could not be removed'
        mkdir -- "$lock" 2>/dev/null || fail 'another install is running'
    fi
    lock_owned=1
    printf '%s\n' "$$" > "$lock/pid"
    legacy_count=0
    v2_count=0
    for name in vrhino vrhino-wan-family-convert; do
        dest=$bin_dir/$name
        [ -e "$dest" ] || continue
        [ -f "$dest" ] && [ ! -L "$dest" ] || fail "unmanaged command: $dest"
        if grep -Fqx '# VRhino installer launcher v1' "$dest"; then
            legacy_count=$((legacy_count + 1))
        elif grep -Fqx '# VRhino distribution v2 launcher' "$dest"; then
            v2_count=$((v2_count + 1))
        else
            fail "unmanaged command: $dest"
        fi
    done
    if [ "$legacy_count" -eq 1 ] && [ "$v2_count" -eq 1 ]; then
        [ "$(readlink "$prefix/current" 2>/dev/null || :)" = 'legacy/v0.9.1-alpha' ] ||
            fail 'mixed launcher generations without legacy activation'
        for name in vrhino vrhino-wan-family-convert; do
            saved=$prefix/legacy/v0.9.1-alpha/bin/$name
            [ -f "$saved" ] && grep -Fqx '# VRhino installer launcher v1' "$saved" ||
                fail 'legacy launcher backup missing'
            if grep -Fqx '# VRhino installer launcher v1' "$bin_dir/$name"; then
                cmp -s "$saved" "$bin_dir/$name" || fail 'legacy launcher backup mismatch'
            else
                render_launcher "$name" | cmp -s - "$bin_dir/$name" ||
                    fail 'mixed launcher does not match this v2 store'
            fi
        done
        for name in vrhino vrhino-wan-family-convert; do
            cp -p -- "$prefix/legacy/v0.9.1-alpha/bin/$name" "$bin_dir/$name.restore-$$" ||
                fail 'legacy launcher recovery failed'
            mv -Tf -- "$bin_dir/$name.restore-$$" "$bin_dir/$name" ||
                fail 'legacy launcher recovery failed'
        done
        legacy_count=2
        v2_count=0
    fi
    if [ "$legacy_count" -eq 0 ] && [ "$v2_count" -eq 1 ]; then
        [ -L "$prefix/current" ] || fail 'partial v2 launcher without active core'
        "$prefix/current/bin/vrhino" --version >/dev/null || fail 'partial v2 active core unavailable'
        for name in vrhino vrhino-wan-family-convert; do
            dest=$bin_dir/$name
            if [ -f "$dest" ]; then
                render_launcher "$name" | cmp -s - "$dest" || fail 'partial v2 launcher mismatch'
            else
                render_launcher "$name" > "$dest.restore-$$"
                chmod 755 "$dest.restore-$$"
                mv -Tf -- "$dest.restore-$$" "$dest" || fail 'partial v2 launcher recovery failed'
            fi
        done
        v2_count=2
    fi
    [ "$legacy_count" -eq 0 ] || [ "$legacy_count" -eq 2 ] || fail 'incomplete legacy launcher pair'
    [ "$v2_count" -eq 0 ] || [ "$v2_count" -eq 2 ] || fail 'incomplete v2 launcher pair'
    [ "$legacy_count" -eq 0 ] || [ "$v2_count" -eq 0 ] || fail 'mixed launcher generations'
    if [ "$v2_count" -eq 2 ]; then
        [ -L "$prefix/current" ] || fail 'v2 launcher has no active version'
    fi
    if [ "$legacy_count" -eq 2 ]; then
        "$bin_dir/vrhino" --version | grep -Fq 'v0.9.1-alpha' || fail 'legacy launcher is not v0.9.1-alpha'
        [ ! -e "$prefix/current" ] ||
            [ "$(readlink "$prefix/current" 2>/dev/null || :)" = 'legacy/v0.9.1-alpha' ] ||
            fail 'legacy launcher conflicts with current v2 activation'
        legacy_migration=1
    fi
    mkdir -p -- "$prefix/runtime" "$prefix/dependencies"
    printf 'vrhino-distribution-v2\n' > "$prefix/.vrhino-v2-store"
    work=$(mktemp -d "$prefix/.stage.XXXXXXXX")
    verify_tree() {
        tree=$1
        (cd "$tree" && sha256sum --status -c SHA256SUMS) || fail "internal checksum failed: $tree"
        tab=$(printf '\t')
        while IFS="$tab" read -r path target; do
            [ -n "$path" ] || continue
            [ -L "$tree/$path" ] || fail "missing symlink: $path"
            [ "$(readlink "$tree/$path")" = "$target" ] || fail "symlink mismatch: $path"
            case "$target" in /*|../*|*/../*|*/..|..) fail "unsafe symlink: $path";; esac
        done < "$tree/SYMLINKS.tsv"
    }
    archive_for() {
        name=$1; hash=$2; url=$3
        if [ -n "$bundle" ]; then
            file=$bundle/$name
            [ -f "$file" ] || fail "artifact missing: $name"
        else
            [ -n "$url" ] || fail "URL missing for $name"
            file=$work/$name
            curl_fetch "$url" "$file"
        fi
        actual=$(sha256sum < "$file"); actual=${actual%% *}
        [ "$actual" = "$hash" ] || fail "outer SHA mismatch: $name"
        echo "$file"
    }
    runtime_path=$prefix/dependencies/$runtime_id
    if [ -d "$runtime_path" ]; then
        [ -f "$runtime_path/.artifact-sha256" ] &&
            [ "$(cat "$runtime_path/.artifact-sha256")" = "$runtime_sha" ] || fail 'installed runtime archive identity mismatch'
        verify_tree "$runtime_path"
        echo "Reusing NVIDIA runtime: $runtime_id"
    else
        runtime_archive=$(archive_for "$runtime_name" "$runtime_sha" "$runtime_url")
        tar -xzf "$runtime_archive" -C "$work" --no-same-owner || fail 'runtime extraction failed'
        [ -d "$work/$runtime_root" ] && [ ! -L "$work/$runtime_root" ] || fail 'runtime archive layout mismatch'
        verify_tree "$work/$runtime_root"
        grep -Fqx "id=$runtime_id" "$work/$runtime_root/dependency.txt" || fail 'runtime identity mismatch'
        grep -Fqx 'runtime_api=1' "$work/$runtime_root/dependency.txt" || fail 'runtime API mismatch'
        printf '%s\n' "$runtime_sha" > "$work/$runtime_root/.artifact-sha256"
        mv -T -- "$work/$runtime_root" "$runtime_path" || fail 'runtime activation failed'
    fi
    grep -Fqx "id=$runtime_id" "$runtime_path/dependency.txt" || fail 'runtime identity mismatch'
    grep -Fqx 'runtime_api=1' "$runtime_path/dependency.txt" || fail 'runtime API mismatch'
    core_path=$prefix/runtime/$core_version
    if [ -d "$core_path" ]; then
        [ -f "$core_path/.artifact-sha256" ] &&
            [ "$(cat "$core_path/.artifact-sha256")" = "$core_sha" ] || fail 'installed core archive identity mismatch'
        verify_tree "$core_path"
    else
        core_archive=$(archive_for "$core_name" "$core_sha" "$core_url")
        tar -xzf "$core_archive" -C "$work" --no-same-owner || fail 'core extraction failed'
        [ -d "$work/$core_root" ] && [ ! -L "$work/$core_root" ] || fail 'core archive layout mismatch'
        verify_tree "$work/$core_root"
        grep -Fqx "id=$runtime_id" "$work/$core_root/requires-nvidia.txt" || fail 'core requires another runtime'
        grep -Fqx "sha256=$runtime_sha" "$work/$core_root/requires-nvidia.txt" || fail 'core runtime SHA mismatch'
        grep -Fqx 'runtime_api=1' "$work/$core_root/requires-nvidia.txt" || fail 'core runtime API mismatch'
        printf '%s\n' "$core_sha" > "$work/$core_root/.artifact-sha256"
        mv -T -- "$work/$core_root" "$core_path" || fail 'core staging activation failed'
    fi
    grep -Fqx "id=$runtime_id" "$core_path/requires-nvidia.txt" || fail 'core requires another runtime'
    grep -Fqx "sha256=$runtime_sha" "$core_path/requires-nvidia.txt" || fail 'core runtime SHA mismatch'
    grep -Fqx 'runtime_api=1' "$core_path/requires-nvidia.txt" || fail 'core runtime API mismatch'
    [ -f "$core_path/bin/vrhino" ] || fail 'core launcher missing'
    "$core_path/bin/vrhino" --version >/dev/null || fail 'core preflight failed'
    "$core_path/bin/vrhino" doctor >/dev/null || fail 'doctor preflight failed'
    if [ "$legacy_migration" -eq 1 ]; then
        legacy_path=$prefix/legacy/v0.9.1-alpha
        if [ -d "$legacy_path" ]; then
            for name in vrhino vrhino-wan-family-convert; do
                cmp -s "$bin_dir/$name" "$legacy_path/bin/$name" || fail 'legacy backup mismatch'
            done
        else
            mkdir -p -- "$work/legacy-v0.9.1-alpha/bin" "$prefix/legacy"
            for name in vrhino vrhino-wan-family-convert; do
                cp -p -- "$bin_dir/$name" "$work/legacy-v0.9.1-alpha/bin/$name" || fail 'legacy backup failed'
                cmp -s "$bin_dir/$name" "$work/legacy-v0.9.1-alpha/bin/$name" || fail 'legacy backup verification failed'
            done
            mv -T -- "$work/legacy-v0.9.1-alpha" "$legacy_path" || fail 'legacy backup activation failed'
        fi
        "$legacy_path/bin/vrhino" --version | grep -Fq 'v0.9.1-alpha' || fail 'legacy rollback target unavailable'
        if [ ! -e "$prefix/current" ]; then
            ln -s 'legacy/v0.9.1-alpha' "$work/legacy-current"
            mv -Tf -- "$work/legacy-current" "$prefix/current" || fail 'legacy current initialization failed'
            legacy_current_created=1
        fi
    fi
    for name in vrhino vrhino-wan-family-convert; do
        dest=$bin_dir/$name
        launcher=$work/$name
        render_launcher "$name" > "$launcher"
        chmod 755 "$launcher"
        if [ "$v2_count" -eq 2 ]; then
            cmp -s "$launcher" "$dest" || fail "v2 launcher points to another installation: $dest"
        fi
    done
    if [ "$legacy_migration" -eq 0 ] && [ "$v2_count" -eq 0 ]; then
        ln -s "runtime/$core_version" "$work/current"
        mv -Tf -- "$work/current" "$prefix/current" || fail 'initial activation failed'
        initial_current_created=1
    fi
    if [ "$v2_count" -eq 0 ]; then
        for name in vrhino vrhino-wan-family-convert; do
            mv -T -- "$work/$name" "$bin_dir/$name" || fail 'command installation failed'
            installed_commands="$installed_commands $name"
        done
    fi
    # Both stable commands now resolve through current. One rename switches the active version.
    ln -s "runtime/$core_version" "$work/current"
    mv -Tf -- "$work/current" "$prefix/current" || fail 'atomic activation failed'
    activation_committed=1
    echo "Activated $core_version; NVIDIA runtime $runtime_id"
)
install_main "$@"
