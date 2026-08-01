#!/bin/bash
# --- MI50 (GFX906) MMQ nwarps sweep for mx-llama.cpp, ROCm 6.3.4 ---
#
# Automates the plan in plans/gfx906-mmq-nwarps-tuning.md: for each (Q8, OTHER) nwarps
# combination it reconfigures + rebuilds with your normal MI50 flags plus the two
# GGML_MMQ_NWARPS_GFX906_* defines, runs the correctness gate (test-backend-ops -o MUL_MAT,
# to catch the q5_1 flicker), benchmarks with llama-bench, and tabulates avg t/s.
#
# Based on the standard MI50 build script (RCCL=ON is mandatory for TP perf, ROCWMMA_FATTN
# stays OFF on gfx906, LTO on). Uses a SEPARATE build dir so your main build/ is untouched,
# and ccache so per-config rebuilds only recompile the MMQ objects.
#
# Usage:  scripts/bench-gfx906-nwarps.sh [MODEL_PATH]
# Tunables are the env vars in the CONFIG block below.
#
# Default runs single-GPU on the 32GB card (device 1). TP (-sm tensor) load OOMs on the 16GB
# card for large models (device 0 = 16368 MiB, the even split gives it ~half the weights + KV
# + compute buffers -> the GGML_ASSERT meta_buf... crash). To bench multi-GPU TP anyway, set
# BENCH_ENV="HIP_VISIBLE_DEVICES=0,1" and BENCH_ARGS="... -sm tensor -ts 1,2 ...".
set -uo pipefail

[[ ! -f "CMakeLists.txt" ]] && echo "❌ Nicht im llama.cpp-Root-Verzeichnis!" && exit 1

# ------------------------------------------------------------------ CONFIG ---
MODEL="${1:-${MODEL:-$HOME/data/models/Qwopus3.6-35B-A3B-Coder-APEX-MTP-Balanced.gguf}}"# For a Q4_0 model (e.g. Ornith-1.0-9B-Q4_0.gguf) use:
#   MODEL=~/data/models/Ornith-1.0-9B-Q4_0.gguf OTHER_VALUES="4 8 16" BENCH_ARGS="..." scripts/bench-gfx906-nwarps.sh
# because Q4_0 is governed by GGML_MMQ_NWARPS_GFX906_OTHER, not Q8.BUILD_DIR="${BUILD_DIR:-build-nwarps-sweep}"     # separate from your main build/

# Sweep matrix (space-separated warp counts per knob). Use POWERS OF 2 only — non-pow2 values
# (e.g. 12) don't divide the MMQ tiles and fail the MUL_MAT gate.
# NOTE: the Q8 knob only affects Q8_0-weight models; the OTHER knob affects every other quant
# (Q4_K/Q5_K/Q6_K/IQ*/…). Sweep the dimension that matches your model's quant — for a non-Q8
# model, fix Q8_VALUES="8" and sweep OTHER_VALUES.
Q8_VALUES="${Q8_VALUES:-8}"
OTHER_VALUES="${OTHER_VALUES:-4 8 16}"

# Correctness gate:
GATE_RUNS="${GATE_RUNS:-2}"
SKIP_GATE="${SKIP_GATE:-0}"

# Benchmark (minus model + -o). Default: single GPU on the 32GB card (device 1), no tensor
# split — avoids the TP-load OOM on the 16GB card and isolates the MMQ kernel perf cleanly.
# The model must fit on that one card; use a smaller/more-quantized model if it doesn't (the
# nwarps effect shows on any quantized model). For multi-GPU TP instead:
#   BENCH_ENV="HIP_VISIBLE_DEVICES=0,1" BENCH_ARGS="... -sm tensor -ts 1,2 ..."
BENCH_ENV="${BENCH_ENV:-HIP_VISIBLE_DEVICES=1}"
BENCH_ARGS="${BENCH_ARGS:--ngl 99 -fa 1 -mmp 0 -dio 1 -r 3 -p 512 -n 0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-bench-nwarps-$STAMP}"
# -----------------------------------------------------------------------------

# --- toolchain env (same as the MI50 build script) ---
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}
export AMDGPU_ARCH="gfx906"

if command -v ccache &>/dev/null; then
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  export CMAKE_HIP_COMPILER_LAUNCHER=ccache
fi

GENERATOR="Unix Makefiles"
command -v ninja &>/dev/null && GENERATOR="Ninja"

RESULTS="$OUTDIR/results.tsv"
BENCH_BIN="$BUILD_DIR/bin/llama-bench"
GATE_BIN="$BUILD_DIR/bin/test-backend-ops"

[ -f "$MODEL" ] || { echo "❌ Modell nicht gefunden: $MODEL"; exit 1; }
mkdir -p "$OUTDIR"
printf 'q8\tother\tgate\tresults(avg±sd t/s)\n' > "$RESULTS"

echo "== gfx906 MMQ nwarps sweep =="
echo "model:   $MODEL"
echo "build:   $BUILD_DIR ($GENERATOR, ccache: $(command -v ccache >/dev/null && echo yes || echo no))"
echo "sweep:   Q8={$Q8_VALUES}  OTHER={$OTHER_VALUES}"
echo "bench:   $BENCH_ENV llama-bench $BENCH_ARGS"
echo "out:     $OUTDIR/"
echo

# Full configure with the MI50 flags + the two nwarps defines. CMake is idempotent, so after
# the first run only the changed defines trigger a (partial, MMQ-only) recompile.
configure() {
  local q8="$1" other="$2" log="$3"
  cmake -B "$BUILD_DIR" -S . -G "$GENERATOR" \
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
    -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_CXX_FLAGS="-Wno-unused-command-line-argument" \
    -DGGML_MMQ_NWARPS_GFX906_Q8="$q8" \
    -DGGML_MMQ_NWARPS_GFX906_OTHER="$other" >"$log" 2>&1
}

run_one() {
  local q8="$1" other="$2"
  local tag="q8=${q8} other=${other}"
  local log="$OUTDIR/build_q8-${q8}_other-${other}.log"
  echo "---------------------------------------------------------------"
  echo ">> $tag : configure + build"

  if ! configure "$q8" "$other" "$log"; then
    echo "   cmake FAILED (see $log)"; printf '%s\t%s\tCONFIG_FAIL\t-\n' "$q8" "$other" >>"$RESULTS"; return
  fi
  if ! cmake --build "$BUILD_DIR" --target llama-bench test-backend-ops -j "$(nproc)" >>"$log" 2>&1; then
    echo "   build FAILED (see $log)"; printf '%s\t%s\tBUILD_FAIL\t-\n' "$q8" "$other" >>"$RESULTS"; return
  fi

  # correctness gate
  local gate="SKIPPED"
  if [ "$SKIP_GATE" != "1" ]; then
    local pass=0
    for ((g=1; g<=GATE_RUNS; g++)); do
      env $BENCH_ENV "$GATE_BIN" -o MUL_MAT >"$OUTDIR/gate_q8-${q8}_other-${other}_run${g}.log" 2>&1 && pass=$((pass+1))
    done
    gate="${pass}/${GATE_RUNS}"
    echo "   MUL_MAT gate: $gate"
    [ "$pass" -ne "$GATE_RUNS" ] && echo "   !! gate nicht sauber — Config als verdächtig behandeln"
  fi

  # benchmark (json for reliable parsing)
  echo ">> $tag : benchmark"
  local jout="$OUTDIR/bench_q8-${q8}_other-${other}.json"
  local errf="$OUTDIR/bench_q8-${q8}_other-${other}.err"
  if ! env $BENCH_ENV "$BENCH_BIN" -m "$MODEL" $BENCH_ARGS -o json >"$jout" 2>"$errf"; then
    echo "   llama-bench FAILED (see $errf)"
    if grep -q "failed to load model" "$errf" 2>/dev/null; then
      echo "   -> Modell passt nicht in den sichtbaren VRAM. Kleineres/stärker quantisiertes"
      echo "      Modell nehmen, oder Multi-GPU: BENCH_ENV=\"HIP_VISIBLE_DEVICES=0,1\""
      echo "      BENCH_ARGS=\"... -sm tensor -ts 1,2 ...\""
    fi
    printf '%s\t%s\t%s\tBENCH_FAIL\n' "$q8" "$other" "$gate" >>"$RESULTS"; return
  fi

  local summary
  summary="$(python3 - "$jout" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
out = []
for r in rows:
    npr, ngn = r.get("n_prompt", 0), r.get("n_gen", 0)
    lbl = f"pp{npr}" if ngn == 0 else (f"tg{ngn}" if npr == 0 else f"pp{npr}+tg{ngn}")
    a, s = r.get("avg_ts"), r.get("stddev_ts")
    out.append(f"{lbl}={a:.2f}±{s:.2f}" if a is not None else f"{lbl}=?")
print("  ".join(out))
PY
)"
  echo "   -> $summary"
  printf '%s\t%s\t%s\t%s\n' "$q8" "$other" "$gate" "$summary" >>"$RESULTS"
}

for q8 in $Q8_VALUES; do
  for other in $OTHER_VALUES; do
    run_one "$q8" "$other"
  done
done

echo
echo "== summary =="
column -t -s $'\t' "$RESULTS"
echo
echo "results: $RESULTS"
echo "Wähle die höchsten Werte mit sauberem (N/N) Gate UND bestem pp, trage sie als Defaults"
echo "in ggml/src/ggml-cuda/mmq.cuh ein und dokumentiere sie in docs/gfx906-optimization-notes.md."
