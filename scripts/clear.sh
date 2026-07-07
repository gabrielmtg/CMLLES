#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/env.sh"

usage() {
    echo "Usage: $0 [--all | --libs | --benchmarks | --image]"
    echo ""
    echo "  --benchmarks   Remove compiled benchmark executables (forces recompile)"
    echo "  --libs         Remove cross-compiled libraries (forces recompile)"
    echo "  --image        Remove the built SD card image (keeps libs and executables)"
    echo "  --all          Remove everything (libs + executables + image + sysroot)"
    echo ""
    echo "Without flags: removes executables + image (keeps downloaded RPiOS and libs)."
}

if [ $# -eq 0 ]; then
    echo "Clearing benchmark executables and built image..."
    rm -rf "${EXEC_DIR}"
    rm -f  "${IMAGES_DIR}/cmlles-raspios.img"
    echo "Done. Run ./scripts/build.sh to rebuild."
    exit 0
fi

for arg in "$@"; do
    case "$arg" in
        --benchmarks)
            echo "Clearing benchmark executables..."
            rm -rf "${EXEC_DIR}"
            ;;
        --libs)
            echo "Clearing cross-compiled libraries..."
            rm -rf "${LIBS_PREFIX}"
            rm -rf "${OUTPUT_DIR}/bookworm-sysroot"
            rm -f  "${OUTPUT_DIR}/glibc_compat.o"
            ;;
        --image)
            echo "Clearing built SD card image..."
            rm -f "${IMAGES_DIR}/cmlles-raspios.img"
            ;;
        --all)
            echo "Clearing all build artifacts..."
            rm -rf "${EXEC_DIR}"
            rm -rf "${LIBS_PREFIX}"
            rm -rf "${OUTPUT_DIR}/bookworm-sysroot"
            rm -f  "${OUTPUT_DIR}/glibc_compat.o"
            rm -f  "${IMAGES_DIR}/cmlles-raspios.img"
            echo "Note: the original RPiOS download (${IMAGES_DIR}/${RPIOS_FILENAME}) was kept."
            echo "      Remove it manually if you also want to re-download it."
            break
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

echo "Done. Run ./scripts/build.sh to rebuild."
