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
- [x] Risiko: mittel-hoch (zentraler Pfad)
- [x] Geschätzter Aufwand: 1–2 Wochen
- [ ] **Verworfen:** Kompletter Ersatz von `mmq.cuh` durch skyne98s Version, da zu stark an dessen neues MMQ-Subsystem gekoppelt

### 2.3 MMVQ-Kernel (Matrix-Vektor)
- [ ] `gfx906/matmul/mmvq-q4_0.cuh` portieren
- [ ] `gfx906/matmul/mmvq-q4_1.cuh` portieren
- [ ] `gfx906/matmul/mmvq-q8_0.cuh` portieren
- [ ] Routing für token-generation (kleines `n`) einbauen
- [ ] Benchmark-Fokus: tg128/tg512
- [ ] Risiko: mittel
- [ ] Geschätzter Aufwand: 1 Woche

---

## Phase 3: Mittel-Hoch – GEMM/FP16/FP32

### 3.1 `sgemm.cuh` (FP32 GEMM)
- [ ] Portieren und in `ggml_cuda_mul_mat_cublas` als alternativen Pfad einbinden
- [ ] Testfälle identifizieren, bei denen hipBLAS `CUBLAS_STATUS_INTERNAL_ERROR` wirft
- [ ] Benchmark: FP32-MUL_MAT-Fälle aus `test-backend-ops`
- [ ] Risiko: mittel-hoch
- [ ] Geschätzter Aufwand: 1 Woche

### 3.2 `mmf.cuh` (FP16 GEMM)
- [ ] Portieren des tiled 32×64×64 FP16-GEMM
- [ ] Einbinden für Batch-Größen 9–2048
- [ ] Benchmark: größere Batch-Größen (pp2048/pp4096/pp8192)
- [ ] Risiko: mittel-hoch
- [ ] Geschätzter Aufwand: 1–2 Wochen

---

## Phase 4: Hoch – Flash Attention mit Q8-Cache

### 4.1 `fattn-q8.cuh` + `fattn-q8.cu`
- [ ] Header und Wrapper portieren
- [ ] Tile-Scheduling an aktuelles `flash-attn-common.cuh` anpassen
- [ ] Unterstützte DKQ/DV-Kombinationen prüfen
- [ ] Risiko: hoch (komplex, viele Template-Instanzen)
- [ ] Geschätzter Aufwand: 2–3 Wochen

### 4.2 `instances/*.cu`
- [ ] Für jede DKQ/DV-Kombination Instanzen erzeugen
- [ ] Build-Zeit beobachten (viele Template-Instanziierungen)
- [ ] Risiko: mittel
- [ ] Geschätzter Aufwand: 3–5 Tage

### 4.3 Integration & Tests
- [ ] Routing in `ggml_cuda_flash_attn_ext` einbauen
- [ ] Test mit `--cache-type-k q8_0 --cache-type-v q8_0`
- [ ] Vergleich: `-fa 1` mit/ohne gfx906-Flash-Attention
- [ ] Benchmark: große Kontexte (32768/65536/131072)
- [ ] Risiko: hoch
- [ ] Geschätzter Aufwand: 1 Woche

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
