#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: build-media-lip-sync.sh FFMPEG_TAR_XZ X264_TAR_GZ SOXR_SYSROOT WORK_DIR PREFIX" >&2
    exit 2
fi

ffmpeg_archive=$(realpath "$1")
x264_archive=$(realpath "$2")
soxr_sysroot=$(realpath "$3")
work_dir=$(realpath -m "$4")
prefix=$(realpath -m "$5")
logical_prefix=/opt/vrhino-media

if ! command -v nasm >/dev/null 2>&1; then
    echo "nasm is required to build the qualified x86 assembly path" >&2
    exit 2
fi
if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -print -quit)" ]]; then
    echo "work directory must be empty: $work_dir" >&2
    exit 2
fi
if [[ ! -f "$soxr_sysroot/usr/include/soxr.h" ]]; then
    echo "SOXR_SYSROOT must contain usr/include/soxr.h" >&2
    exit 2
fi

mkdir -p "$work_dir" "$prefix"
install_root="$work_dir/install-root"
tar -xJf "$ffmpeg_archive" -C "$work_dir"
tar -xzf "$x264_archive" -C "$work_dir"

ffmpeg_source=$(find "$work_dir" -mindepth 1 -maxdepth 1 -type d -name 'ffmpeg-*' -print -quit)
x264_source=$(find "$work_dir" -mindepth 1 -maxdepth 1 -type d -name 'x264-*' -print -quit)
test -n "$ffmpeg_source"
test -n "$x264_source"

(
    cd "$x264_source"
    ./configure \
        --prefix="$logical_prefix" \
        --enable-shared \
        --disable-cli \
        --disable-opencl

    # A git archive does not carry .git metadata, so x264's generated version
    # header otherwise omits the audited revision. Restore only that fixed
    # source identity; this does not change encoder computation or defaults.
    sed -i \
        -e 's/^#define X264_VERSION .*/#define X264_VERSION " r3060 5db6aa6"/' \
        -e 's/^#define X264_POINTVER .*/#define X264_POINTVER "0.163.3060 5db6aa6"/' \
        x264_config.h

    make -j2
    make DESTDIR="$install_root" install
)

x264_pc="$install_root$logical_prefix/lib/pkgconfig"
soxr_include="$soxr_sysroot/usr/include"
soxr_lib="$soxr_sysroot/usr/lib/x86_64-linux-gnu"

(
    cd "$ffmpeg_source"
    PKG_CONFIG_SYSROOT_DIR="$install_root" \
    PKG_CONFIG_PATH="$x264_pc" \
    ./configure \
        --prefix="$logical_prefix" \
        --disable-everything \
        --disable-autodetect \
        --disable-network \
        --disable-doc \
        --disable-debug \
        --disable-static \
        --enable-shared \
        --enable-ffmpeg \
        --disable-ffprobe \
        --disable-ffplay \
        --enable-avcodec \
        --enable-avformat \
        --enable-avfilter \
        --enable-swscale \
        --enable-swresample \
        --enable-libsoxr \
        --enable-zlib \
        --disable-avdevice \
        --disable-postproc \
        --enable-protocol=file,pipe \
        --enable-demuxer=rawvideo,mov,wav,matroska,image2,image_png_pipe \
        --enable-decoder=rawvideo,h264,pcm_s16le,aac,ffv1,png \
        --enable-parser=h264,aac \
        --enable-muxer=mp4,null,rawvideo,pcm_f32le,wav \
        --enable-filter=scale,format,aresample,aformat,pan \
        --enable-encoder=libx264,wrapped_avframe,rawvideo,pcm_f32le,aac \
        --enable-libx264 \
        --enable-gpl \
        --extra-cflags="-I$install_root$logical_prefix/include -I$soxr_include" \
        --extra-ldflags="-L$install_root$logical_prefix/lib -L$soxr_lib"
    make -j2
    make DESTDIR="$install_root" install
)

cp -a "$install_root$logical_prefix"/. "$prefix"/
cp -L "$soxr_lib/libsoxr.so.0" "$prefix/lib/libsoxr.so.0"
ln -sfn libsoxr.so.0 "$prefix/lib/libsoxr.so"

patchelf --set-rpath '$ORIGIN/../lib' "$prefix/bin/ffmpeg"
for library in \
    "$prefix"/lib/libavfilter.so.*.*.* \
    "$prefix"/lib/libavformat.so.*.*.* \
    "$prefix"/lib/libavcodec.so.*.*.* \
    "$prefix"/lib/libswresample.so.*.*.* \
    "$prefix"/lib/libswscale.so.*.*.* \
    "$prefix"/lib/libavutil.so.*.*.* \
    "$prefix"/lib/libx264.so.*; do
    patchelf --set-rpath '$ORIGIN' "$library"
done

"$prefix/bin/ffmpeg" -version
"$prefix/bin/ffmpeg" -buildconf
