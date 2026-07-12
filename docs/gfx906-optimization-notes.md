# gfx906 (MI50 / MI60) Optimization Notes

Distilled reference for kernel/backend work on this fork's target hardware (AMD Instinct
MI50/MI60, LLVM target `gfx906`). Condensed from the measured studies at
<https://github.com/skyne98/wiki-gfx906> (data cross-checked 2026-02-21 on real 4× gfx906
hosts, 60 CUs @ 1725 MHz). Numbers are microbenchmark-derived — directional, not contractual.

---

## 1. Architecture baseline

| Spec | MI60 | MI50 (16/32 GB) |
|---|---|---|
| LLVM target | `gfx906` | `gfx906` |
| CUs | 64 | 60 |
| Wavefront | 64 lanes | 64 lanes |
| LDS/CU | 64 KiB (32 banks × 4 B) | 64 KiB (32 banks × 4 B) |
| L2 (shared) | 4 MiB | 4 MiB |
| L1 vector/CU | 16 KiB, write-through, 64 B lines | same |
| VGPR / SGPR file | 256 KiB / 12.5 KiB per CU | same |
| Peak FP16 / FP32 / FP64 | 29.5 / 14.8 / 7.4 TFLOPS | 26.8 / 13.4 / 6.7 TFLOPS |
| HBM2 BW / TDP | up to 1 TB/s / 300 W | same |
| Interconnect | 2× Infinity Fabric (xGMI), up to 200 GB/s P2P; PCIe Gen4 x16 up to 64 GB/s | same |

- Treat MI50 and MI60 as the **same ISA family**; only CU count and VRAM differ.
- CU = 4× SIMD16; up to 40 resident wavefronts/CU, but **VGPR/SGPR/LDS pressure gates
  occupancy before nominal wave-slot maxima** in most real kernels.

## 2. ISA / compiler surface (important constraints)

- **Compile explicitly for the right XNACK mode** — do not leave it ambiguous across
  environments: `--offload-arch=gfx906:xnack-` or `gfx906:xnack+`. (`sramecc` is **not**
  available on gfx906.)
- **No MFMA on gfx906.** `v_mfma*` is a gfx908+ feature. Do **not** assume matrix-core paths.
- **No newer dependency-control instructions:** `s_clause`, `s_waitcnt_depctr`, `s_delay_alu`
  are all rejected by the gfx906 assembler. Latency control = ILP + occupancy + careful
  `s_waitcnt` placement only.
- `wavefrontsize64` is the mode for this generation (workgroups in multiples of 64).

## 3. Dot instructions (the gfx906 "DL" path)

Available and validated on gfx906:

| Intrinsic | Instruction | Semantics |
|---|---|---|
| `llvm.amdgcn.sdot4` / `udot4` | `v_dot4_i32_i8` / `_u32_u8` | 2× packed i32 (4×8-bit) → 32-bit accum |
| `llvm.amdgcn.sdot8` / `udot8` | `v_dot8_i32_i4` / `_u32_u4` | 2× packed i32 (8×4-bit) → 32-bit accum |
| `llvm.amdgcn.fdot2` | `v_dot2_f32_f16` | 2× packed f16 → f32 accum |

- **`dot4c` / `dot8c` are NOT available** — only the non-`c` forms.
- Accumulator is **32-bit and can overflow**; the 4th intrinsic operand is clamp-enable.
  Clamp saturates (`INT_MAX`/`INT_MIN`/`UINT_MAX`); no-clamp wraps. Enable clamp only when
  saturation is actually required.
- **Throughput** (ILP4, ~81% of theoretical peak): `sdot4` ~43–44 TOPS, `sdot8` ~85–86 TOPS.
  **dot8 ≈ 2× dot4** — prefer dot8 when the quant layout supports 4-bit packing.
- **Keep multiple independent accumulators** per thread — dependency-chained single-accum
  runs roughly halve achieved throughput (~22 TOPS sdot4 vs ~44 with ILP4).

## 4. Quant/dequant helper instructions (all validated on gfx906)

- **SDWA** (`*_sdwa`, all 239 GFX9 SDWA mnemonics assemble on gfx906): byte/word select
  directly inside ALU/convert ops — ideal for i8 dequant
  (`v_cvt_f32_i32_sdwa ... src0_sel:BYTE_{0..3}`). Selects sublanes only (`BYTE_0..3`,
  `WORD_0..1`, `DWORD`), not arbitrary bitfields.
- **Bitfield/pack:** `v_bfe_i32` (nibble extract + sign-extend, for int4), `v_lshl_or_b32`,
  `v_perm_b32` (byte permute), `v_med3_i32` (saturating clamp to a range, e.g. [-128,127]).
- **Packed fp16:** `v_pack_b32_f16`, `v_cvt_pkrtz_f16_f32` (pack+convert f32→2×f16),
  `v_pk_add/mul/fma_f16` (2 lanes/op).
- **Wave-lane movement without LDS:** `v_mov_b32_dpp`, `ds_bpermute_b32` / `ds_permute_b32`.
  These are wave-level only, not cross-wave/global.

## 5. FP32 activations: quantize or not?

Measured, hot-loop conversion (activations start FP32, 8 MAC/thread/iter):

| Path | TOPS |
|---|---|
| `pure_fp32` | ~5.95 |
| `qdq_fp16_dot2` | ~4.19 |
| `qdq_int8_dot4` | ~2.00 |

- **If conversion happens per-use → pure FP32 wins.** The pack/convert cost dominates and
  the dot paths lose.
- **If conversion is amortized over many reuses (GEMM-like):** `fp32_reuse` ~13 TOPS vs
  `dot4_reuse`/`dot2_reuse` ~21.7–21.9 TOPS — dot paths win. **Decide by reuse depth**, not
  by theoretical dot peak.

## 6. LDS layout standard for LLM tiles

- gfx906 LDS = 32 banks × 4 B. Column-style reads with **power-of-two leading dimension
  alias banks and collapse bandwidth.**
- Measured (`ds_read_b128`): contiguous vec4 ~4257 GB/s; column-style `ld=32` vec4
  ~1865 GB/s; **same with `ld=33` (one vec4 pad) ~3974 GB/s** — the pad recovers most BW.
- **Rule:** row-consumed operand (A) → natural stride; column-consumed operand (B) →
  `stride_vec = K_vec + 1` (one vec4 pad per row). Overhead ~`1/K_vec`, usually worth it.
- Use 16-byte vectorized, 16-byte-aligned LDS payloads and prefer `ds_read/write_b128`.

## 7. KV-cache layout for decode kernels

- Canonical layout = **`HSD` = `[head][seq][dim]`** (dim contiguous).
- Writes: `HSD` ~357 GB/s vs `HDS` ~14–54 GB/s → **HSD hugely better for token writes.**
- Dot-style decode reads (per-seq dot over dim): `HSD` ~1.76 TB/s vs `HDS` ~0.37 TB/s.
- Only exception: a kernel that is explicitly **dim-fixed streaming over seq** can prefer
  `HDS` (~73.7 vs ~41.5 GB/s) — specialized, not a default.
- Use `x4` (`global_load/store_dwordx4`) vectorized access when aligned.

## 8. Latency hiding (measured)

- **Row-local lane shuffle → `v_mov_b32_dpp`** (`row_shr` etc.): ~2× an LDS+barrier
  equivalent (~1780 vs ~906 Gxchg/s), no barrier.
- **Arbitrary in-wave shuffle → `ds_bpermute_b32` / `ds_permute_b32`**: beats LDS
  store+load+barrier (~965 vs ~906 Gxchg/s).
- **LDS staging → `ds_read/write_b128`**: b128 (~9.5–11.2 TB/s) ≫ b64 (~4.3–8.8) ≫ b32
  (~1.9–3.9). Default to b128 where alignment allows.
- **Global staging → `global_load_dwordx4`** for contiguous packed data (~870 vs ~814 GB/s).
- **Structure loops to issue several independent loads before first use**; don't wait
  immediately after each load. Staged `s_waitcnt vmcnt(...)`/`lgkmcnt(...)` is the core
  latency-hiding mechanism here (no `s_clause`/`s_delay_alu` available).

## 9. Kernel-tuning checklist (default starting point)

1. Compile with explicit `--offload-arch=gfx906:xnack{-,+}`.
2. Workgroups in multiples of 64; sweep block sizes watching VGPR/LDS pressure.
3. Keep LDS bank-friendly (the `+1` vec4 pad rule for column-consumed tiles).
4. Coalesce global access; prefer `dwordx4` loads and `b128` LDS.
5. KV-cache in `HSD`; decode math around per-seq dot traversal.
6. Prefer documented `v_dot*` / SDWA / packed-fp16 paths; **never assume MFMA**.
7. Multi-GPU: verify real Infinity-Fabric topology before optimizing collectives for P2P.
8. Only quantize FP32 activations when conversion is amortized by high reuse.

## Primary sources

- ROCm GPU arch specs: <https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>
- HIP hardware implementation: <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html>
- LLVM AMDGPU usage: <https://llvm.org/docs/AMDGPUUsage.html>
- LLVM gfx906 asm syntax: <https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html> (contrast gfx908 for MFMA)
- AMD Vega 7nm Shader ISA: <https://gpuopen.com/wp-content/uploads/2019/11/Vega_7nm_Shader_ISA_26November2019.pdf>
- Study source repo: <https://github.com/skyne98/wiki-gfx906>
