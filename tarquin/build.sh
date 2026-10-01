#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
####################################################################################################
#                                             build.sh                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Builds a portable command-line TARQUIN from the pinned upstream commit plus             #
#          cli-build.patch, on the machine's own OS and architecture: macOS (needs nothing         #
#          beyond the OS), Linux (fully static, e.g. inside Alpine) or Windows (MSYS2 UCRT64,      #
#          static). Boost and FFTW are built here as static libraries from checksummed sources;    #
#          BLAS/LAPACK is Accelerate on macOS and reference LAPACK, built here, elsewhere.         #
#          Needs cmake, ninja or make, a C/C++ compiler, gfortran (or $FC), curl, git and tar.     #
#          Output: dist/tarquin (dist/tarquin.exe on Windows).                                     #
#                                                                                                  #
####################################################################################################
set -euo pipefail

TARQUIN_REPO=https://github.com/martin3141/tarquin.git
TARQUIN_REF=47e9b98263bf27f5a732445b1a13121005326b09     # 2021-07-22, reports 4.3.11
TARQUIN_VERSION=4.3.11
BOOST_VERSION=1.92.0
BOOST_SHA256=9bed76128d4e46755dbe818487788c6fceb6f72b378f4daa49b7e1e600d9088d
FFTW_VERSION=3.3.11
FFTW_SHA256=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
LAPACK_VERSION=3.12.1
LAPACK_SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422

case "$(uname -s)" in
    Darwin)       os=macos ;;
    Linux)        os=linux ;;
    *_NT*)        os=windows ;;   # MSYS2 (MINGW64_NT, UCRT64_NT, MSYS_NT, ...)
    *)            echo "unsupported system: $(uname -s)" >&2; exit 1 ;;
esac
exe="tarquin$([ "$os" = windows ] && echo .exe || true)"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$os" = windows ]; then
    here="$(cygpath -m "$here")"   # the native toolchain reads D:/..., not MSYS's /d/...
fi
work="$here/work"
deps="$work/deps"
dist="$here/dist"
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
mkdir -p "$work/downloads" "$dist"

# On macOS, C and C++ from Apple's clang so libc++ is the system one; elsewhere the toolchain
# on PATH. Fortran (cvmlib, and LAPACK off macOS) is gfortran unless FC names another.
if [ "$os" = macos ]; then
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
    export SDKROOT="$(xcrun --show-sdk-path)"
    export CC="$(xcrun -f clang)" CXX="$(xcrun -f clang++)"
else
    export CC="${CC:-gcc}" CXX="${CXX:-g++}"
fi
FC="${FC:-gfortran}"

common=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX")
if command -v ninja > /dev/null; then
    common+=(-G Ninja)
fi


#***********#
#   fetch   #
#***********#
fetch() {   # url sha256 -> path of the verified download
    local file="$work/downloads/${1##*/}"
    [ -f "$file" ] || { curl -fsSL --retry 3 -o "$file.part" "$1" && mv "$file.part" "$file"; }
    local got
    got="$( (sha256sum "$file" 2>/dev/null || shasum -a 256 "$file") | cut -d' ' -f1)"
    [ "$got" = "$2" ] || { echo "sha256 mismatch for $1: $got" >&2; exit 1; }
    echo "$file"
}


#--------------------------------------------------------------------------------------------------#
#  FFTW, static, plain C (no SIMD: a CI runner's vector units are not the user's)                  #
#--------------------------------------------------------------------------------------------------#
if [ ! -f "$deps/lib/libfftw3.a" ]; then
    rm -rf "$work/fftw-$FFTW_VERSION"
    tar -xzf "$(fetch "https://www.fftw.org/fftw-$FFTW_VERSION.tar.gz" "$FFTW_SHA256")" -C "$work"
    (cd "$work/fftw-$FFTW_VERSION" \
        && ./configure --prefix="$deps" --disable-shared --enable-static --disable-fortran \
                       --disable-dependency-tracking CC="$CC" CFLAGS=-O3 \
        && make -j"$jobs" && make install)
fi


#--------------------------------------------------------------------------------------------------#
#  Boost, static, only the libraries TARQUIN links                                                 #
#--------------------------------------------------------------------------------------------------#
if [ ! -f "$deps/lib/libboost_filesystem.a" ]; then
    rm -rf "$work/boost-$BOOST_VERSION" "$work/boost-build"
    tar -xJf "$(fetch "https://github.com/boostorg/boost/releases/download/boost-$BOOST_VERSION/boost-$BOOST_VERSION-cmake.tar.xz" "$BOOST_SHA256")" -C "$work"
    cmake -S "$work/boost-$BOOST_VERSION" -B "$work/boost-build" "${common[@]}" \
        -DCMAKE_INSTALL_PREFIX="$deps" -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF -DCMAKE_CXX_STANDARD=14 \
        -DBOOST_INCLUDE_LIBRARIES="date_time;filesystem;system;thread"
    cmake --build "$work/boost-build"
    cmake --install "$work/boost-build"
fi


#--------------------------------------------------------------------------------------------------#
#  reference BLAS and LAPACK, static (macOS has Accelerate)                                        #
#--------------------------------------------------------------------------------------------------#
if [ "$os" != macos ] && [ ! -f "$deps/lib/liblapack.a" ]; then
    rm -rf "$work/lapack-$LAPACK_VERSION" "$work/lapack-build"
    tar -xzf "$(fetch "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v$LAPACK_VERSION.tar.gz" "$LAPACK_SHA256")" -C "$work"
    cmake -S "$work/lapack-$LAPACK_VERSION" -B "$work/lapack-build" "${common[@]}" \
        -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_INSTALL_PREFIX="$deps" -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DCBLAS=OFF -DLAPACKE=OFF \
        -DBUILD_INDEX64_EXT_API=OFF
    cmake --build "$work/lapack-build"
    cmake --install "$work/lapack-build"
fi


#--------------------------------------------------------------------------------------------------#
#  TARQUIN                                                                                         #
#--------------------------------------------------------------------------------------------------#
src="$work/tarquin"
rm -rf "$src"
git init -q "$src"
git -C "$src" fetch -q --depth 1 "$TARQUIN_REPO" "$TARQUIN_REF"
git -C "$src" -c advice.detachedHead=false checkout -q FETCH_HEAD
git -C "$src" apply "$here/cli-build.patch"

# -std=gnu++14: the code predates C++17; directory.hpp: newer Boost no longer pulls it in
# through operations.hpp; finite (macOS): levmar's C uses the BSD name the SDK has dropped,
# while glibc, musl and MinGW still declare it.
# -L deps/lib holds only archives, so -lfftw3 (and -lblas/-llapack off macOS) are static;
# on macOS -lblas/-llapack resolve to Accelerate. Static reference LAPACK needs BLAS after
# it (cvmlib names BLAS first, and a static link resolves in order) and the Fortran
# runtime, which the standard libraries put last on the link line.
flags=(-DCMAKE_CXX_FLAGS="-w -std=gnu++14 -include boost/filesystem/directory.hpp -I$deps/include")
if [ "$os" = macos ]; then
    flags+=(-DCMAKE_C_FLAGS="-w -Dfinite=isfinite"
            -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"
            -DCMAKE_EXE_LINKER_FLAGS="-L$deps/lib")
else
    flags+=(-DCMAKE_C_FLAGS="-w")
    runtime="-lblas -lgfortran"
    [ "$("$FC" -print-file-name=libquadmath.a)" != libquadmath.a ] && runtime+=" -lquadmath"
    link="-static -L$deps/lib"
    # musl gives each thread 128 KB of stack, where glibc gives 8 MB; TARQUIN simulates its
    # internal basis in threads that need more, and musl takes the default from this
    if [ "$os" = linux ]; then
        link+=" -Wl,-z,stack-size=8388608"
    fi
    flags+=(-DCMAKE_EXE_LINKER_FLAGS="$link"
            -DCMAKE_CXX_STANDARD_LIBRARIES="$runtime")
fi
cmake -S "$src/src" -B "$src/build" "${common[@]}" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_PREFIX_PATH="$deps" "${flags[@]}"
cmake --build "$src/build" --target tarquin

cp "$src/build/redist/$exe" "$dist/$exe"
strip "$dist/$exe"
if [ "$os" = macos ]; then
    codesign --force --sign - "$dist/$exe"
fi


#--------------------------------------------------------------------------------------------------#
#  portability check: the binary alone, in an empty folder, linking only the OS                    #
#--------------------------------------------------------------------------------------------------#
check="$(mktemp -d)"
cp "$dist/$exe" "$check/"
case "$os" in
    macos)
        otool -l "$check/$exe" | grep -A4 LC_BUILD_VERSION | grep -E 'minos|sdk'
        for lib in $(otool -L "$check/$exe" | tail -n +2 | awk '{print $1}'); do
            case "$lib" in
                /usr/lib/libSystem.B.dylib | /usr/lib/libc++.1.dylib | /System/Library/Frameworks/*) ;;
                *) echo "not an OS library: $lib" >&2; exit 1 ;;
            esac
        done ;;
    linux)
        if readelf -d "$check/$exe" | grep -q NEEDED; then
            readelf -d "$check/$exe" >&2; echo "not statically linked" >&2; exit 1
        fi ;;
    windows)
        # system DLLs are KERNEL32.dll, api-ms-win-*.dll and the like; MinGW's runtime
        # (libstdc++-6, libgcc_s, libwinpthread-1, libgfortran-5, ...) all start with 'lib'
        objdump -p "$check/$exe" | grep 'DLL Name'
        if objdump -p "$check/$exe" | grep 'DLL Name' | grep -qi 'DLL Name: lib'; then
            echo "links a MinGW runtime DLL" >&2; exit 1
        fi ;;
esac
status=0
(cd "$check" && "./$exe" --help > help.txt 2>&1) || status=$?
grep -q "TARQUIN $TARQUIN_VERSION Started" "$check/help.txt" && [ "$status" -eq 255 ] \
    || { cat "$check/help.txt" >&2; echo "--help exited $status" >&2; exit 1; }
rm -rf "$check"
echo "portable: $dist/$exe ($os $(uname -m))"
