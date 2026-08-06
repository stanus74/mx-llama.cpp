#!/usr/bin/env bash
# Build script for gfx906 / MI50 (Vega20), ROCm 6.x.
#
# Incremental by default. Use --clean only when the CMake cache is actually stale
# (changed toolchain, changed cmake flags, moved directory). A full wipe costs
# minutes even with ccache, because reconfiguring invalidates two thirds of it.
#
# Flag choices are measured, not guessed:
#   GGML_HIP_RCCL=ON       required for -sm tensor throughput. Without it: ~720
#                          instead of ~1000 t/s pp512, bisected over five commits.
#   GGML_HIP_ROCWMMA_FATTN gfx906 has no matrix cores (CDNA/gfx908+ only). The
#                          rocwmma-dev package may be installed, it buys nothing here.
#   GGML_HIP_NO_VMM=ON     this stack reports VMM: no.
#   GGML_HIP_GRAPHS=ON     graph capture is a clear win.
#   GGML_LTO=ON            no measured downside.
set -uo pipefail

CLEAN=0
CHECK=0
JOBS=$(nproc)

usage() {
    cat <<'USAGE'
Aufruf: scripts/mx-compile.sh [OPTIONEN]

  --clean       build/ komplett neu anlegen (sonst inkrementell)
  --check       nach dem Bau test-backend-ops -o MUL_MAT laufen lassen
  -j N          Parallelitaet (Vorgabe: nproc)
  -h, --help    diese Hilfe
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=1; shift ;;
        --check) CHECK=1; shift ;;
        -j)      JOBS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unbekannte Option: $1"; usage; exit 2 ;;
    esac
done

[[ -f CMakeLists.txt && -d ggml ]] || { echo "❌ Nicht im llama.cpp-Wurzelverzeichnis"; exit 1; }

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="$ROCM_PATH"
export HIP_PLATFORM=amd
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}"

ARCH=gfx906
LOG="build_mx_mi50_$(date +%Y%m%d_%H%M%S).log"   # matched by /build* in .gitignore

echo "🚀 Ziel: MI50 ($ARCH) · ROCm $("$ROCM_PATH/bin/hipconfig" --version 2>/dev/null || echo '?')"

if command -v ccache >/dev/null; then
    echo "✓ ccache"
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_HIP_COMPILER_LAUNCHER=ccache
fi

if command -v ninja >/dev/null; then
    GENERATOR=Ninja; BUILD=(ninja -j "$JOBS" -C build)
else
    GENERATOR="Unix Makefiles"; BUILD=(make -j "$JOBS" -C build)
fi

if (( CLEAN )); then
    echo "🧹 build/ wird neu angelegt"
    rm -rf build
fi

# Reconfigure only when there is no cache yet. This is what turns a config-only
# change into a seconds-long rebuild instead of a multi-minute one.
if [[ ! -f build/CMakeCache.txt ]]; then
    echo "⚙  CMake-Konfiguration" | tee -a "$LOG"
    cmake -B build -G "$GENERATOR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
        -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
        -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
        -DAMDGPU_TARGETS="$ARCH" \
        -DGPU_TARGETS="$ARCH" \
        -DGGML_HIP=ON \
        -DGGML_HIP_GRAPHS=ON \
        -DGGML_HIP_NO_VMM=ON \
        -DGGML_HIP_RCCL=ON \
        -DGGML_HIP_ROCWMMA_FATTN=OFF \
        -DGGML_LTO=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_CXX_FLAGS="-Wno-unused-command-line-argument" \
        2>&1 | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "❌ CMake fehlgeschlagen, siehe $LOG"; exit 1; }
else
    echo "⚙  vorhandene CMake-Konfiguration wird weiterverwendet (--clean erzwingt neu)"
fi

echo "🔨 Bau mit -j$JOBS" | tee -a "$LOG"
SECONDS=0
"${BUILD[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}

if (( STATUS != 0 )); then
    echo "❌ Build fehlgeschlagen nach ${SECONDS}s — Fehler aus $LOG:"
    grep -iE "error:" "$LOG" | head -5
    exit 1
fi
echo "✅ Build erfolgreich in ${SECONDS}s"

if (( CHECK )); then
    echo "🔍 test-backend-ops -o MUL_MAT (gefiltert)"
    # An unfiltered run aborts at MUL_MAT(type_a=f32) with CUBLAS_STATUS_INTERNAL_ERROR
    # on this ROCm version - pre-existing, untouched mainline does the same. Filter by type.
    for t in q4_K q5_K q6_K q8_0; do
        n=$(HSA_XNACK=0 HIP_VISIBLE_DEVICES=1 build/bin/test-backend-ops -o MUL_MAT 2>/dev/null \
            | grep -c "type_a=$t.*OK")
        printf '  %-6s %s bestanden\n' "$t" "$n"
    done
fi

cat <<'RUN'

── Laufzeit-Umgebung ────────────────────────────────────────────────────
  export HSA_XNACK=0                    # PFLICHT: sonst melden sich die Karten
                                        # als xnack+, rocBLAS hat dafuer keinen
                                        # Sgemm-Kernel und MoE-Modelle stuerzen
                                        # beim Prefill ab
  export HSA_OVERRIDE_GFX_VERSION=9.0.6
  export HSA_P2P_DISABLE=1              # X99: kein PLX-Switch zwischen den
  export HIP_FORCE_P2P_DISABLE=1        # MI50-Root-Ports, P2P haengt sonst

  ⚠ Nicht-interaktives SSH liest ~/.bashrc NICHT. In Skripten und ueber
    'ssh host cmd' die Variablen explizit setzen, sonst misst man etwas
    anderes als im Terminal. Kontrolle: env | grep -i hsa

── Betrieb ──────────────────────────────────────────────────────────────
  Multi-GPU, schnellste gemessene Variante (+82% pp / +51% tg gegen eine Karte):

    HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-server -m MODELL.gguf \
      -ngl 99 -fa 1 -sm tensor -lm dio --host 0.0.0.0 --port 8080

  Immer -lm dio: mmap haengt auf diesem Stack. Die alten Schreibweisen
  --no-mmap / -dio sind veraltet und laufen in llama-bench doppelt.

  llama-cli ist seit b10240 eine Chat-UI: fuer Skripte -no-cnv -st </dev/null,
  sonst laeuft sie ohne TTY endlos weiter.
RUN
