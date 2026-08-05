# hipBLAS `Sgemm` stürzt am MoE-Router ab (gfx906, ROCm 6.3.4)

**Gefunden:** 2026-08-05 · **Status:** Ursache eingegrenzt, Workaround verifiziert

Qwen3.6-35B-A3B-Modelle brechen auf dieser Maschine beim ersten Decode ab:

```
ROCm error: CUBLAS_STATUS_INTERNAL_ERROR
  current device: 0, in function ggml_cuda_mul_mat_cublas_impl at ggml/src/ggml-cuda/ggml-cuda.cu:1557
  hipblasSgemm(ctx.cublas_handle(), HIPBLAS_OP_T, HIPBLAS_OP_N, ne01, ne11, ne10, …)
```

## Workaround (sofort nutzbar)

```bash
export GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
```

Verifiziert auf `Qwopus3.6-35B-A3B-Coder-APEX-MTP-Balanced`, MI50, eine GPU:
**pp512 764,84 t/s, tg32 51,84 t/s** — ohne die Variable Absturz.

⚠ Der Schalter ist **global**: er stellt *jeden* cuBLAS-Matmul auf F16-Compute um, nicht nur den
betroffenen. Für den Router ist das unkritisch, für andere F32-Pfade eine Genauigkeitsänderung.
Als Dauerlösung gehört die Fallunterscheidung in den Code, nicht in die Umgebung.

## Ursache

Die abstürzende Operation ist der **MoE-Router** (`ffn_gate_inp`). Der ist F32, geht deshalb nicht
über MMQ, sondern über `ggml_cuda_mul_mat_cublas` → `hipblasSgemm`. hipBLAS dieser ROCm-Version
verträgt die Form nicht.

Tensorformen je Modell (`gguf-py`, F32-Tensoren mit ≥2 Dimensionen):

| Modell | `ffn_gate_inp` | Ergebnis |
|---|---|---|
| `Qwopus3.6-35B-A3B-Coder-APEX-MTP-Balanced` | **[2048, 256]** | Absturz |
| `Ornith-1.0-35B-Heretic-MTP-APEX-I-Balanced` | **[2048, 256]** | Absturz |
| `gemma-4-26B-A4B-it-UD-Q6_K_XL` | [2816, 128] | läuft |
| `Qwopus3.6-27B-Coder-Compat-MTP-Q6_K` | *kein Router (dicht)* | läuft |

Es ist also **formabhängig**, nicht MoE-abhängig: gemmas Router mit 128 Experten geht durch, der
mit 256 nicht.

## Nur Prefill, nicht Generierung

Entscheidend für die Frage „das lief doch schon mal": **Token-Generierung ist nicht betroffen.**

| Aufruf | Batch | Ergebnis (`Ornith-1.0-35B`) |
|---|---|---|
| `-p 0 -n 32` | 1 | **läuft, 50,80 t/s** |
| `-p 128` | 128 | Absturz |
| `-p 512` | 512 | Absturz |
| `-p 2048` | 2048 | Absturz |

Bei einem einzelnen Token ist der Router ein Matvec und geht über `mmvf`; erst ab mehreren Tokens
wird daraus ein echtes GEMM und landet in `hipblasSgemm`. Ein Modell kann sich also im Alltag
lange unauffällig verhalten — Generierung läuft, kurze Prompts fallen nicht auf — und erst beim
Prefill eines längeren Prompts umfallen.

## Was ausgeschlossen wurde

Jede dieser Möglichkeiten wurde einzeln geprüft und widerlegt:

- **Der GCN5-MMQ-Patch.** Der betrifft nur den MMQ-Pfad für quantisierte Gewichte; der Router ist
  F32 und läuft an MMQ vorbei.
- **Der q8_1-Cache.** Mit `GGML_CUDA_Q8_1_CACHE=0` identischer Absturz — geprüft für *beide*
  betroffenen Modelle, nicht nur eines.
- **Eine Regression durch Mainline.** Der alte Fork (`445cf9bdd`, `/opt/mx-llama.cpp`) stürzt an
  derselben Stelle ab (`ggml_cuda_mul_mat_cublas`).
- **Speicherdruck / Hintergrundprozess.** GPU nachweislich leer (10 MB belegt von 32 GB), keine
  llama-Prozesse; Absturz unverändert.
- **GPU-Anzahl.** Mit einer wie mit zwei Karten identisch.

## Zusammenhang mit Schritt 3.1 des GCN5-Plans

Dies ist dieselbe Fehlerklasse, die ein *ungefilterter* `test-backend-ops -o MUL_MAT`-Lauf bei
`MUL_MAT(type_a=f32)` zeigt und die dort gegen unverändertes Mainline als vorbestehendes
hipBLAS-Problem eingestuft wurde. Neu ist: **es ist kein Test-Artefakt.** Der Defekt macht reale
Modelle unbenutzbar.

## Offene Fäden

- Welche Dimension genau kippt es — die 256 (Expertenzahl) oder die 2048? Ein Sweep über
  `test-backend-ops`-Formen würde das klären.
- Tritt es unter neueren ROCm-Versionen noch auf? gfx906 ist ab ROCm 7.x deprecated, ein Update
  ist also kein sicherer Ausweg.
- Saubere Lösung wäre, kleine F32-Matmuls dieser Form nicht an hipBLAS zu geben. Das wäre ein
  Upstream-Beitrag mit klarem Reproduzierer.
