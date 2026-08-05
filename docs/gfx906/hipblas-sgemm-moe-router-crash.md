# `HSA_XNACK=0` fehlt → hipBLAS `Sgemm` stürzt am MoE-Router ab (gfx906)

**Gefunden:** 2026-08-05 · **Status:** Ursache geklärt, Umgebungsfehler

Qwen3.6-35B-A3B-Modelle brechen beim Prefill ab:

```
ROCm error: CUBLAS_STATUS_INTERNAL_ERROR
  current device: 0, in function ggml_cuda_mul_mat_cublas_impl at ggml/src/ggml-cuda/ggml-cuda.cu:1557
  hipblasSgemm(ctx.cublas_handle(), HIPBLAS_OP_T, HIPBLAS_OP_N, ne01, ne11, ne10, …)
```

## Ursache: `HSA_XNACK`

Die Karte meldet sich je nach Umgebung mit **zwei verschiedenen Code-Object-Targets**:

| Umgebung | Gerätekennung | Prefill |
|---|---|---|
| `HSA_XNACK=0` | `gfx906:sramecc+:**xnack-**` | **läuft** |
| `HSA_XNACK` nicht gesetzt | `gfx906:sramecc+:**xnack+**` | **Absturz** |

rocBLAS hat für `xnack+` offenbar keine passenden `Sgemm`-Kernel für diese Form und wirft
`CUBLAS_STATUS_INTERNAL_ERROR`. Mit `xnack-` läuft dieselbe Operation ohne Umweg.

`~/.bashrc` auf `x99` setzt das korrekt (Zeile 142):

```bash
export HSA_XNACK=0    # Deaktiviert Memory-Retry (wichtig für Performance)
```

Verifiziert, sonst identischer Aufruf, MI50, eine GPU:

| Modell | pp512 mit `HSA_XNACK=0` | ohne |
|---|---:|---|
| `Qwopus3.6-35B-A3B-Coder-APEX-MTP-Balanced` | **771,51 t/s** | Absturz |
| `Ornith-1.0-35B-Heretic-MTP-APEX-I-Balanced` | **772,66 t/s** | Absturz |

## ⚠ Falle: nicht-interaktive SSH-Sitzungen

**`ssh host 'befehl'` liest die `.bashrc` nicht.** Ein Kommando, das im Login-Shell einwandfrei
läuft, landet über SSH in einer völlig anderen GPU-Umgebung — hier fehlten `HSA_XNACK`,
`HSA_OVERRIDE_GFX_VERSION`, `HSA_P2P_DISABLE` und `HIP_FORCE_P2P_DISABLE` allesamt.

Für reproduzierbare Messungen über SSH die Variablen **explizit mitgeben** oder
`source ~/.bashrc` voranstellen. Ein `env | grep -i hsa` am Anfang einer Messreihe hätte den
gesamten Irrweg unten erspart.

## Was dabei fälschlich als Ursache dokumentiert wurde

Die erste Fassung dieses Dokuments erklärte den Absturz als **formabhängigen hipBLAS-Fehler** am
MoE-Router (`ffn_gate_inp`, F32, 2048×256) und empfahl `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` als
Workaround. Das war falsch — oder genauer: es beschrieb korrekt, *wo* es knallt, aber nicht *warum*.

Die Beobachtungen von damals bleiben gültig und erklären sich jetzt zwanglos:

- **Nur die 35B-A3B-Modelle betroffen, gemma-26B-A4B nicht.** Formabhängig ist es tatsächlich —
  aber nur, weil unter `xnack+` bestimmte Kernel fehlen, nicht weil hipBLAS die Form generell
  nicht kann.
- **Nur Prefill, nicht Generierung.** Bei Batch 1 ist der Router ein Matvec und geht über `mmvf`,
  also gar nicht durch rocBLAS.
- **`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` half.** Es weicht auf einen anderen rocBLAS-Kernel aus, der
  auch unter `xnack+` existiert. Es ist damit **überflüssig und obendrein langsamer**: 756 t/s
  gegen 771 t/s mit korrekter Umgebung — bei gleichzeitig schlechterer Genauigkeit für *alle*
  cuBLAS-Matmuls. **Nicht verwenden.**
- **Alter Fork stürzte genauso ab.** Klar — er lief in derselben kaputten SSH-Umgebung.

## Lehre

Der Fehler wurde über mehrere Runden immer präziser eingegrenzt — Modell, Tensorform, Batch-Größe —
und lag die ganze Zeit außerhalb des untersuchten Bereichs. Alle Kontrollmessungen (q8_1-Cache aus,
alter Fork, GPU leer, eine vs. zwei Karten) waren korrekt und haben korrekt ausgeschlossen, was sie
ausschließen sollten; sie liefen nur alle in derselben falschen Umgebung.

**Konsequenz für künftige Messungen:** die Umgebung gehört zum Messaufbau und muss protokolliert
werden, nicht nur die Kommandozeile.
