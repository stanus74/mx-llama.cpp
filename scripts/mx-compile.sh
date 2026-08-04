#!/bin/bash
# --- MI50 (GFX906) mx-llama.cpp FINAL BUILD SCRIPT, ROCm 6.3.4 ---
# Erkenntnisse aus dem Debugging-Marathon:
#   - GGML_HIP_RCCL=ON ist PFLICHT fuer TP-Performance (ohne RCCL: ~720 statt
#     ~1000 t/s pp512, verifiziert per Bisect ueber 5 Commits, kein Code-Bug)
#   - GGML_HIP_ROCWMMA_FATTN bleibt IMMER OFF: gfx906/Vega20 hat keine
#     Matrix-Cores (die gibt es erst ab CDNA/gfx908+), das Paket rocwmma-dev
#     kann installiert sein, bringt auf dieser Architektur aber nichts
#   - TP (-sm tensor) + MTP-Draft crasht mit ROCm OOM auf der kleineren
#     16GB-Karte (CPU-Sampler-Fallback: "backend sampling not supported
#     with SPLIT_MODE_TENSOR"). Fuer MTP-Modelle Layer-Split nutzen, nicht TP.
#   - Custom-AR (GGML_ENABLE_CUSTOM_AR) bringt auf diesem X99-Board nichts:
#     kein PLX-Switch zwischen den MI50-Root-Ports, P2P daher nicht
#     verfuegbar -> faellt automatisch auf RCCL zurueck. RCCL-Link bleibt
#     trotzdem noetig (siehe oben), nur die Custom-AR-Env-Vars sind optional.
set -e

[[ ! -f "CMakeLists.txt" ]] && echo "❌ Nicht im llama.cpp-Root-Verzeichnis!" && exit 1

export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}

export AMDGPU_ARCH="gfx906"
echo "🚀 Ziel-Architektur: MI50 ($AMDGPU_ARCH) — mx-llama.cpp, RCCL+TP-optimiert"

if command -v ccache &> /dev/null; then
  echo "✓ ccache aktiviert"
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  export CMAKE_HIP_COMPILER_LAUNCHER=ccache
fi

GENERATOR="Unix Makefiles"
BUILD_CMD="make -j$(nproc)"
if command -v ninja &> /dev/null; then
  echo "✓ Ninja gefunden, nutze Ninja statt Make (schneller)"
  GENERATOR="Ninja"
  BUILD_CMD="ninja"
fi

rm -rf build && mkdir -p build && cd build

LOG_FILE="../build_mx_mi50_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logs werden gespeichert in: $LOG_FILE"

echo "=== CMAKE CONFIGURATION (mx-llama.cpp, RCCL+TP, LTO) ===" | tee -a "$LOG_FILE"
cmake .. -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
  -DAMDGPU_TARGETS="$AMDGPU_ARCH" \
  -DGPU_TARGETS="$AMDGPU_ARCH" \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DGGML_LTO=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CXX_FLAGS="-Wno-unused-command-line-argument" \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== BUILDING FOR MI50 (mx-llama.cpp, RCCL+LTO aktiv) ===" | tee -a "$LOG_FILE"
$BUILD_CMD 2>&1 | tee -a "$LOG_FILE"
BUILD_STATUS=${PIPESTATUS[0]}

if [ $BUILD_STATUS -eq 0 ]; then
  echo "✅ Build erfolgreich!" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
  echo "  Layer-Split + MTP (fuer MTP/Speculative-Decoding-Modelle):" | tee -a "$LOG_FILE"
  echo "    HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-server -m MODELL.gguf -ngl 99 -fa 1 --no-mmap -dio --spec-type draft-mtp -md DRAFT.gguf" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
  echo "  Tensor-Parallel, KEIN MTP (fuer reine PP-lastige Workloads):" | tee -a "$LOG_FILE"
  echo "    HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-server -m MODELL.gguf -ngl 99 -fa 1 -sm tensor --no-mmap -dio" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
  echo "  ⚠ TP + MTP zusammen NICHT verwenden: crasht mit ROCm OOM auf der 16GB-Karte" | tee -a "$LOG_FILE"
else
  echo "❌ Build fehlgeschlagen! Checke $LOG_FILE"
  exit 1
fi
