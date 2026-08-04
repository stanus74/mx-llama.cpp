# Portierungsplan: gfx906-Kernel aus skyne98/llama.cpp-gfx906

**Quelle:** [skyne98/llama.cpp-gfx906](https://github.com/skyne98/llama.cpp-gfx906) Branch `ruquant-w4a4`  
**Ziel:** `mx-llama.cpp` (aktueller Merge-Branch `merge-upstream-20260801`)  
**Zielplattform:** AMD ROCm / gfx906 (MI50/MI60)  
**Sortierung:** Einfach → Schwierig  
**Geschätzte Gesamtdauer:** 2–6 Wochen bei sequenzieller Arbeit

---

## Vorbereitung (Voraussetzungen für alle folgenden Schritte)

- [ ] Verzeichnis `ggml/src/ggml-cuda/gfx906/` anlegen
- [ ] Gemeinsame Dateien portieren:
  - [ ] `gfx906-config.h` (Tuning-Konstanten, MMQ_NWARPS, Flags)
  - [ ] `gfx906-common.cuh` (load-Makros, dp4a, Warp-Reductions)
- [ ] Build-System anpassen (`CMakeLists.txt` im ggml-cuda-Verzeichnis)
- [ ] Branch `port-skyne98-gfx906` von `merge-upstream-20260801` abzweigen
- [ ] Server-Build-Ziel definieren, um jede Änderung mit `llama-bench` zu verifizieren

---

## Phase 1: Einfach – Grundlagen & niedriges Risiko

### 1.1 RoPE-Optimierung
- [x] `gfx906/attention/rope.cuh` portieren
- [x] In `ggml/src/ggml-cuda/rope.cu` (oder Nachfolger) einklinken
- [x] Verwendung von `__sincosf()` statt getrenntem `sinf()`/`cosf()` testen
- [x] Benchmark: 9B Q5_K_M, pp512/tg128 vorher/nachher vergleichen
  - Vorher: pp512 679.79 t/s, tg128 48.78 t/s
  - Nachher: pp512 675.37 t/s, tg128 50.08 t/s
  - Hinweis: RoPE-Anteil gering, daher marginaler Unterschied
- [x] Risiko: niedrig
- [x] Geschätzter Aufwand: 1–2 Tage

### 1.2 Epilog für Quantisierung
- [ ] `gfx906/quantize/epilogue.cuh` portieren
- [ ] Als Hilfsheader in bestehende `dequantize.cuh`/`quantize.cu` integrieren
- [ ] Risiko: niedrig
- [ ] Geschätzter Aufwand: 1–2 Tage

---

## Phase 2: Mittel – MMQ- und vec_dot-Kernel

### 2.1 `vecdotq.cuh`
- [x] `gfx906/quantize/vecdotq.cuh` portieren
- [x] Gemeinsame Infrastruktur anlegen:
  - [x] `ggml/src/ggml-cuda/gfx906/gfx906-config.h`
  - [x] `ggml/src/ggml-cuda/gfx906/gfx906-common.cuh`
- [x] Unterstütztes Format: MXFP4 (via `__builtin_amdgcn_perm` Lookup)
- [x] Einklinken in `ggml/src/ggml-cuda/vecdotq.cuh` (`vec_dot_mxfp4_q8_1`)
- [x] `test-backend-ops test -o MUL_MAT -p "type_a=mxfp4"` laufen lassen
  - Ergebnis: **44/44 tests passed** auf ROCm0/gfx906
- [x] Regressionstest: 9B Q5_K_M pp512/tg128 unverändert
- [x] Risiko: mittel (Korrektheit muss geprüft werden)
- [x] Geschätzter Aufwand: 3–5 Tage

### 2.2 `mmq.cuh` + `mmq-prefetch.cuh` (chirurgisch)
- [x] `gfx906/matmul/mmq-helpers.cuh` mit vektorisierten Lade-Pfaden anlegen
- [x] Q4_0/Q4_1 y-Tile-Ladung in `vec_dot_q4_0_q8_1_dp4a`/`vec_dot_q4_1_q8_1_dp4a` auf `__gfx906__` vektorisieren
  - Verwendet `int4`-Loads (global_load_dwordx4) statt skalarer `ggml_cuda_memcpy_1`
  - Automatischer Fallback bei nicht aligned Adressen
- [x] `test-backend-ops test -o MUL_MAT -p "type_a=q(4_0|4_1)"`
  - Ergebnis: **91/91 tests passed**
- [x] Regressionstest: 9B Q5_K_M unverändert (Modell nutzt kein Q4_0/Q4_1)
- [ ] **Offen:** Q4_0-Modell (Ornith-1.0-9B-Q4_0) ist unter mainline schneller als im Fork:
  - Fork: pp512 780 t/s, tg128 65.65 t/s
  - Mainline: pp512 842 t/s, tg128 62.24 t/s
  - tg128 ist im Fork schneller (wahrscheinlich MMVQ/nwarps=8), pp512 jedoch langsamer.
  - **Ergebnis Test A (vec-load aus):** pp512 780.65, tg128 66.03 → kein Effekt, Messrauschen.
  - **Ergebnis Test B (nwarps-Sweep):** nwarps=8 war bisher der beste Wert für OTHER.
  - **Ergebnis Test C (`GGML_CUDA_FORCE_CUBLAS=1`):** pp512 778.31, tg128 65.71 → hipBLAS ist für Q4_0 pp512 **nicht** schneller als MMQ.
  - **Ergebnis Test D (`-fa 0`):** pp512 bleibt langsamer im Fork (776 vs 833 mainline) → Ursache nicht in Flash Attention.
  - **Wahrscheinliche Ursache:** Mainline hat ein **redesignetes MMQ-Subsystem** (`ggml_cuda_mmq_get_config_*` pro Architektur), während der Fork den legacy MMQ-Pfad mit gfx906-Tweaks behält. Der legacy-Pfad ist für Q4_0 pp512 strukturell langsamer. Kleinere Tweaks (vec-load, nwarps, hipBLAS-Override) reichen nicht.
  - **Optionen:**
    1. Mainlines neues MMQ-Subsystem portieren (großer Aufwand, Risiko für fork-spezifische Features).
    2. Akzeptieren, dass Q4_0 pp512 im Fork ~7 % langsamer ist, dafür tg128 ~5 % schneller.
    3. Zielgenauere Optimierung des legacy-Pfads für Q4_0 suchen (z. B. Tile-Größe `mmq_y`, Speicherlayout `block_q8_1_mmq`).
    ```bash
    ssh pat@192.168.178.71
    diff -u /opt/llama.cpp/ggml/src/ggml-cuda/mmq.cuh /opt/mx-llama.cpp/ggml/src/ggml-cuda/mmq.cuh > /tmp/diff_mmq.txt
    diff -u /opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu /opt/mx-llama.cpp/ggml/src/ggml-cuda/mmvq.cu > /tmp/diff_mmvq.txt
    diff -u /opt/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu /opt/mx-llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu > /tmp/diff_backend.txt
    diff -u /opt/llama.cpp/src/llama-graph.cpp /opt/mx-llama.cpp/src/llama-graph.cpp > /tmp/diff_graph.txt
    # Kopieren in den lokalen Workspace:
    scp /tmp/diff_*.txt pat@<lokaler-rechner>:/tmp/
    ```
    Alternativ: direkter `rsync` des gesamten `/opt/mx-llama.cpp` und `/opt/llama.cpp` Source-Trees.
- [x] Risiko: mittel-hoch (zentraler Pfad)
- [x] Geschätzter Aufwand: 1–2 Wochen
- [ ] **Verworfen:** Kompletter Ersatz von `mmq.cuh` durch skyne98s Version, da zu stark an dessen neues MMQ-Subsystem gekoppelt

### 2.3 MMVQ-Kernel (Matrix-Vektor)
- [x] `gfx906/matmul/mmvq-q4_0.cuh` portieren
- [x] `gfx906/matmul/mmvq-q4_1.cuh` portieren
- [x] `gfx906/matmul/mmvq-q8_0.cuh` portieren
- [x] Routing für token-generation (kleines `n`) einbauen
- [x] Benchmark-Fokus: tg128/tg512
  - Ergebnis auf Ornith-1.0-9B-Q4_0: **neue Kernel sind langsamer** als existierender MMVQ-Pfad.
    - Vorher (existierender Pfad): tg128 65.65 t/s
    - Nachher (warp-cooperative): tg128 63.32 t/s, tg512 62.46 t/s
  - Ursache: existierender `mul_mat_vec_q`-Pfad für GCN mit `nwarps=2` bei `ncols_dst==1` ist offenbar besser für gfx906 angepasst.
  - Reaktion: `GFX906_MMVQ_WARP_COOP_ENABLED` Schalter hinzugefügt, Default **0** (aus). Code bleibt für weitere Tuning-Experimente verfügbar.
  - Korrektheit: `test-backend-ops test -o MUL_MAT -p "type_a=q(4_0|4_1|8_0)"` → **138/138 passed**.
- [x] Risiko: mittel
- [x] Geschätzter Aufwand: 1 Woche

---

## Phase 3: Mittel-Hoch – GEMM/FP16/FP32

### 3.1 `sgemm.cuh` (FP32 GEMM)
- [ ] Portieren und in `ggml_cuda_mul_mat_cublas` als alternativen Pfad einbinden
- [ ] Testfälle identifizieren, bei denen hipBLAS `CUBLAS_STATUS_INTERNAL_ERROR` wirft
- [ ] Benchmark: FP32-MUL_MAT-Fälle aus `test-backend-ops`
- [ ] Risiko: mittel-hoch
- [ ] Geschätzter Aufwand: 1 Woche

### 3.2 `mmf.cuh` (FP16 GEMM)
- [x] Portieren des tiled 32×64×64 FP16-GEMM
  - Dateien: `gfx906/matmul/sgemm.cuh`, `gfx906/matmul/mmf.cuh`
  - Anpassung: `src1` wird als `float *` gelesen (passt zu `ggml_cuda_mul_mat`-Interface), B-Tile wird beim Laden nach `half` konvertiert.
- [x] Einbinden für Batch-Größen 9–2048 in `ggml_cuda_mul_mat` vor `ggml_cuda_mul_mat_cublas`
- [x] Benchmark: größere Batch-Größen (pp512/pp2048)
  - F16 TinyLlama 1.1B: Kein messbarer Unterschied zwischen gfx906-MMF und hipBLAS/rocBLAS (im Messrauschen).
  - `test-backend-ops test -o MUL_MAT -p "type_a=f16"`: **270/270 passed**.
- [x] Schalter `GFX906_MMF_ENABLED` (Default 1) und Runtime-Env `GGML_CUDA_DISABLE_GFX906_MMF=1` hinzugefügt
- [x] Risiko: mittel-hoch
- [x] Geschätzter Aufwand: 1–2 Wochen
- [x] **Ergebnis:** Kernel korrekt, aber auf getesteter Hardware/Modell nicht schneller. Bleibt als Option verfügbar.

---

## Phase 4: Hoch – Flash Attention mit Q8-Cache

### 4.1 `fattn-q8.cuh` + `fattn-q8.cu`
- [x] Header und Wrapper portieren
- [x] Tile-Scheduling an aktuelles `flash-attn-common.cuh` anpassen
- [x] Unterstützte DKQ/DV-Kombinationen prüfen
  - Aktiv: (64,64), (96,96), (128,128), (256,256), (576,512)
  - Deaktiviert in Instances: (40,40), (80,80), (112,112) — nicht durch 32 teilbar
- [x] Risiko: hoch (komplex, viele Template-Instanzen)
- [x] Geschätzter Aufwand: 2–3 Wochen

### 4.2 `instances/*.cu`
- [x] Für jede DKQ/DV-Kombination Instanzen erzeugen
- [x] Build-System angepasst (`ggml/src/ggml-cuda/CMakeLists.txt` sammelt `gfx906/**/*.cu`)
- [ ] Build-Zeit beobachten (viele Template-Instanziierungen)
- [x] Risiko: mittel
- [x] Geschätzter Aufwand: 3–5 Tage

### 4.3 Integration & Tests
- [x] Routing in `ggml_cuda_flash_attn_ext` einbauen
  - Neuer `best_fattn_kernel` Wert `BEST_FATTN_KERNEL_GFX906_Q8`
  - Compile-Time-Schalter `GFX906_FATTN_Q8_ENABLED` (Default **0**)
  - Runtime-Env `GGML_CUDA_DISABLE_GFX906_FATTN_Q8=1` für A/B
- [x] Test mit `--cache-type-k q8_0 --cache-type-v q8_0`
  - `test-backend-ops test -o FLASH_ATTN_EXT`: Q8_0/Q8_0 Fälle **OK**, einige andere Quant-Typen erwartet "not supported".
- [x] Vergleich: `-fa 1` mit/ohne gfx906-Flash-Attention
  - Q8_0 TinyLlama 1.1B: Kein messbarer Unterschied (pp512 ~4760 t/s, tg128 ~237 vs ~240 t/s — im Rauschen).
  - pp32768: 2035.47 vs 2044.82 t/s (im Rauschen).
  - Ornith-9B-Q4_0 mit Q8_0-KV: pp512 777.50, tg128 64.68 (keine Regression).
- [x] Benchmark: große Kontexte (32768)
- [x] Risiko: hoch
- [x] Geschätzter Aufwand: 1 Woche
- [x] **Abschluss:** Kernel ist korrekt, bringt auf TinyLlama-1.1B-Q8_0 aber keinen messbaren Speedup. `GFX906_FATTN_Q8_ENABLED` daher Default **0**; per `GGML_CUDA_DISABLE_GFX906_FATTN_Q8=0` zum Experimentieren einschaltbar.

---

## Phase 5: Sehr hoch – Fork-spezifische Features

### 5.1 `q8-cache.cuh` (Layer-cycling Q8-Cache)
- [ ] Konzept verstehen (128 MB Slots, Generation-Tracking, Multi-Consumer)
- [ ] Abhängigkeiten zum Graph-Scheduler klären
- [ ] Speicherallokation pro Layer-Gruppe implementieren
- [ ] Test: 35B/70B-Modelle mit großem Kontext bei begrenztem VRAM
- [ ] Risiko: sehr hoch
- [ ] Geschätzter Aufwand: 2–4 Wochen

### 5.2 `graph-fusion.cuh`
- [ ] Multi-Consumer-Analyse im Graph implementieren
- [ ] Voraussetzung für `norm-fused-q8` und `gather-q8`
- [ ] Risiko: sehr hoch
- [ ] Geschätzter Aufwand: 2–3 Wochen

### 5.3 `norm-fused-q8.cu` + `.cuh`
- [ ] Fused RMS-Norm + Q8_1-Quantisierung
- [ ] Graph-Fusion aus 5.2 voraussetzen
- [ ] Risiko: hoch
- [ ] Geschätzter Aufwand: 1–2 Wochen

### 5.4 `gather-q8.cu` + `.cuh`
- [ ] MoE-Row-Gather für Q8_1
- [ ] Nur relevant für MoE-Modelle
- [ ] Risiko: hoch
- [ ] Geschätzter Aufwand: 1 Woche

---

## Test- & Benchmark-Checkliste (gilt für jede Phase)

- [ ] `test-backend-ops test -o MUL_MAT` bestanden für betroffene Formate
- [ ] 9B Q5_K_M single-GPU Benchmark vorher/nachher
- [ ] 27B Q6_K single-GPU Benchmark vorher/nachher
- [ ] 35B Heretic TP-Benchmark (`-sm tensor -tps 2 -fa 1`) vorher/nachher
- [ ] `llama-server` Rauchtest mit kleinem Prompt
- [ ] Keine neuen hipBLAS-/ROCm-Abstürze
- [ ] MERGE_REPORT.md oder separates `gfx906-port.md` aktualisieren

---

## Fazit bisher (Phasen 1–4)

- **Phasen 1 + 2.2** (RoPE, MMQ-vec-load): marginal bis kein realer Gewinn, aber niedriges Risiko.
- **Phase 2.3** (MMVQ warp-cooperative): korrekt, aber langsamer als existierender Pfad → Default aus.
- **Phase 3.2** (MMF FP16 GEMM): korrekt, aber kein Gewinn gegenüber hipBLAS → Default an (kein Nachteil), per Env abschaltbar.
- **Phase 4** (Q8_0 Flash Attention): korrekt, aber kein Gewinn gegenüber upstream tile/vec → Default aus.

**Empfehlung:** Vor Phase 5 (sehr hoher Aufwand) die Optionen prüfen, die vermutlich mehr Potential haben:
1. Phase 3.1 `sgemm.cuh` (FP32-GEMM) für Fälle, in denen hipBLAS fehlschlägt.
2. Vergleich/Diagnose des **legacy MMQ vs. mainline redesign** (Phase 2.2), um den Q4_0-pp512-Nachteil zu verstehen.
3. Phase 5 nur angehen, wenn real VRAM-/Kontext-Engpässe damit gelöst werden sollen.

## Risiken & Abhängigkeiten

| Risiko | Auswirkung | Mitigation |
|---|---|---|
| API-Divergenz zu skyne98s älterem Stand | Build-Fehler / Semantikfehler | Jede Phase isoliert testen, nicht alles auf einmal portieren |
| Kernel-Korrektheit | Falsche Modellausgaben | `test-backend-ops` vor und nach jeder Änderung |
| Build-Zeit-Explosion durch Template-Instanzen | Lange Entwicklungszyklen | Nur relevante DKQ/DV-Kombinationen aktivieren |
| Interaktion mit Multi-Stage TP / MTP | Stille Fehler im Server | Explizite TP-Tests mit großen Modellen |
| Merge-Konflikte bei zukünftigem upstream-Sync | Verlust der Arbeit | Saubere Isolation im `gfx906/`-Unterverzeichnis |

---

## Empfohlener nächster Schritt

Mit **Phase 1.1 (RoPE)** beginnen – geringstes Risiko, schnelle Validierung möglich.
