# Plan: Nächste Optimierungsschritte gfx906 (nach Phasen 1–4)

**Ausgangslage:** [port-gfx906-kernels-from-skyne98.md](port-gfx906-kernels-from-skyne98.md) Phasen 1–4
abgeschlossen. Ergebnis: alle portierten Kernel korrekt, aber **kein messbarer Gewinn** (MMVQ
warp-cooperative langsamer → Default aus; MMF FP16 GEMM im Rauschen; Flash-Attention-Q8 im Rauschen).
Offen und quantifiziert: Q4_0 pp512 im Fork ~7 % langsamer als Mainline (780 vs. 842 t/s).

**Neue Erkenntnis (2026-08-02):** Die Messbasis selbst ist unzuverlässig. Siehe Phase 0.

**Hardware:** 2× MI50 (gfx906), Host `x99` / 192.168.178.71, Build unter `/opt/mx-llama.cpp`.

---

## Phase 0: Messbasis stabilisieren (Voraussetzung für alles Weitere)

**Befund vom 2026-08-02**, 30 Samples über ~60 s Volllast (`rocm-smi`-Loop):

| | GPU0 | GPU1 |
|---|---:|---:|
| Junction max | **99 °C** | 87 °C |
| Edge max | 66 °C | 60 °C |
| **Delta Edge→Junction** | **33 °C** | **27 °C** |
| Memory max | 65 °C | 58 °C |
| Power Ø / max | 175 / 206 W | 174 / 200 W |
| sclk Ø / min unter Last | 1602 / **1485** MHz | 1618 / **1485** MHz |

Interpretation:

- **Delta > 25 °C auf beiden Karten** → Wärmeübergang Die→Kühlkörper defekt. Edge bleibt mit 66 °C
  unauffällig, d. h. der Kühlkörper selbst arbeitet; das Problem sitzt im Interface. Paste ist 6 Monate
  alt → Verdacht Pump-out oder gelockerte Kühlerschrauben. Asymmetrie 33 vs. 27 °C bei identischer
  Vorgeschichte spricht für mechanische Varianz (Anpressdruck).
- **Power-Cap greift korrekt** (PowerPlay-Override via `mi50-tdclimit.service`:
  `TdcLimitGfx=150`, `SocketPowerLimitAc0=180`; `rocm-smi` meldet 180 W Max Package Power) —
  und ist **saturiert** (Ø 175 W).
- **sclk schwankt 1485–1725 MHz (±14 %)** innerhalb einer Minute. Kernel-Effekte von 3–8 % sind
  damit grundsätzlich nicht auflösbar.

**Konsequenz für die Phasen 1–4: Die Aussage „kein Gewinn" ist nicht belastbar.** Sie wurde auf einer
Plattform gemessen, deren Taktrauschen größer war als die gesuchten Effekte.

### Aufgaben

- [ ] Kühlerschrauben nachziehen — über Kreuz, in Stufen, nur bis die Federn auf Block sind
      (Vega20-Die ohne Heatspreader, Überdrehen verbiegt das PCB)
- [ ] Falls Delta danach weiterhin > 20 °C: repasten. Abdruckbild dokumentieren
      (Mitte ausgedünnt + Randwulst = Pump-out bestätigt). Empfehlung PTM7950 statt Paste —
      kann per Definition nicht auspumpen, standfest auf großen heißen Dies
- [ ] HBM-/VRM-Thermalpads beim Zerlegen mitprüfen
- [ ] Messlauf wiederholen, Zielwert: **Delta < 15 °C**
- [ ] Power-Cap schrittweise anheben (200 → 225 W), Junction beobachten
      (`upp set --write smcPPTable/SocketPowerLimitAc0=<W>` in `mi50-tdclimit.service`)
- [ ] Für alle folgenden Benchmarks: `rocm-smi --setperflevel high` setzen, DPM-Sprünge ausschließen

### Abnahmekriterium

sclk unter Last **stabil bei 1725 MHz** über einen kompletten pp2048-Lauf, Junction < 90 °C.
Erst dann ist die Plattform benchmark-tauglich.

### Monitoring-Kommando

```bash
while true; do date +%T; rocm-smi --showtemp --showpower --showclocks; sleep 2; done \
  | tee ~/therm-$(date +%F-%H%M).log
```

---

## Phase A: Bereits geleistete Arbeit nachmessen (billigster Ertrag)

Sobald Phase 0 abgenommen ist. Alle drei Kernel sind fertig und per Schalter A/B-fähig — Aufwand
ist reines Benchmarking, kein Code.

- [ ] **MMF FP16 GEMM:** `GGML_CUDA_DISABLE_GFX906_MMF=1` vs. `=0`, Fokus pp512/pp2048
- [ ] **Flash-Attention Q8:** `GGML_CUDA_DISABLE_GFX906_FATTN_Q8=0` vs. Default,
      Fokus große Kontexte (pp32768) mit `--cache-type-k q8_0 --cache-type-v q8_0`
- [ ] **MMVQ warp-cooperative:** `GFX906_MMVQ_WARP_COOP_ENABLED=1` (Compile-Time), Fokus tg128/tg512
- [ ] Jeweils ≥ 5 Wiederholungen, Median statt Einzelwert, Taktprotokoll mitschreiben
- [ ] Ergebnisse in [port-gfx906-kernels-from-skyne98.md](port-gfx906-kernels-from-skyne98.md)
      nachtragen, Default-Schalter ggf. korrigieren

**Aufwand:** ~1 Tag. **Erwartung:** offen — genau das ist der Punkt.

---

## Phase B: Flag-only-Sweeps (kein Code, hoher Erwartungswert)

Zielt direkt auf den in [docs/gfx906/gfx906-optimization-notes.md](../docs/gfx906/gfx906-optimization-notes.md)
dokumentierten HBM2-Bandbreiten-Engpass.

- [ ] Skript `scripts/bench-gfx906-sweep.sh` analog zu
      [scripts/bench-gfx906-nwarps.sh](../scripts/bench-gfx906-nwarps.sh) anlegen
- [ ] **`-ub`-Sweep:** 256 / 512 / 1024 / 2048 über pp-lastige Workloads
- [ ] **KV-Cache-Quantisierung:** Q8_0 (Baseline) vs. Q4_0 / Q4_1, Fokus tg-Latenz und VRAM
- [ ] Qualitätsprüfung bei KV-Quantisierung: Perplexity gegen Q8_0-Baseline

### B.1 `GGML_MMQ_NWARPS_GFX906_Q8` sweepen — ✅ ERLEDIGT 2026-08-03

**Ergebnis** (Ornith-1.0-9B-Q8_0, pp512, single-GPU auf der 32-GB-Karte, `-ngl 99 -fa 1 -r 3`,
Lauf `bench-nwarps-20260803_105931/`):

| Q8 nwarps | MUL_MAT-Gate | pp512 | Δ vs. 4 |
|---:|---|---|---|
| 4 | 2/2 | 572,07 ±6,13 | — |
| **8 (Default)** | 2/2 | **733,96 ±7,37** | **+28,3 %** |
| 16 | 2/2 | **GPU-Absturz** | — |

Zwei Erkenntnisse:

1. **Default 8 ist für Q8 bestätigt.** +28 % gegenüber 4 bei σ ≈ 6–7 t/s — Effekt ist Faktor 25
   größer als die Streuung. Der offene TODO aus dem `mmq.cuh`-Kommentar ist damit geschlossen.
2. **`nwarps=16` crasht:** `Memory access fault … Write access to a read-only page`.
   Der Vorschlag aus #23881 ("16 could help Q8", gemessen auf MI60) überträgt sich **nicht**.
   → siehe Phase B.2.

**Methodisch wichtig:** Das MUL_MAT-Gate lief bei 16 **sauber 2/2 durch**, der Crash trat erst
in `llama-bench` auf einem echten Modell auf. Das Gate deckt die Shapes realer Modelle nicht ab
und taugt nicht als alleinige Absicherung — dasselbe Muster wie beim `-sm row`-Page-Fault.

**Thermik während des Laufs unkritisch** (Junction max 69 °C, sclk 1606–1725 MHz): pp512-Läufe
sind zu kurz zum Heat-Soak. Phase 0 war für diesen Sweep also nicht blockierend, bleibt es aber
für längere pp4096/pp32768-Läufe.

### B.2 `nwarps=16`-Crash — ⚠️ URSACHE WEITERHIN OFFEN

> **Korrektur 2026-08-03:** Der unten beschriebene Akkumulator-Bug wurde zunächst als Root Cause
> deklariert und behoben (Commit `4e8441f69`). **Mit dem Fix crasht `nwarps=16` unverändert
> weiter** (Re-Sweep auf Ornith-1.0-9B-Q8_0, identischer Fault). Der Bug ist real und der Fix
> korrekt, war aber nicht die Ursache. Die eigentliche Ursache ist unbekannt.

**Gefundener und behobener Bug** (nicht die Crash-Ursache) in [../ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh#L3568):

```c
float sum[mmq_x*mmq_y / (nwarps*warp_size)];
// indiziert als sum[j0/nwarps * mmq_y/warp_size + i0/warp_size]
```

Die Formel setzt implizit `mmq_x >= nwarps` voraus. Auf gfx906 ist `mmq_y=128`, `warp_size=64`,
also `mmq_y/warp_size = 2`; der Dispatcher probiert Kacheln ab `mmq_x=8`:

| nwarps | mmq_x | Array-Größe | benötigte Indizes | |
|---:|---:|---:|---:|---|
| 8 | 8 | 8·128/512 = 2 | 0, 1 | passt exakt |
| 16 | 8 | 8·128/1024 = **1** | 0, 1 | **Off-by-one-Write** |

Die Ganzzahldivision trunkiert → das Array ist ein Element zu kurz → jeder Thread schreibt
darüber hinaus → `Memory access fault … Write access to a read-only page`.

**Folgerungen:**

1. **Shape-abhängig**, deshalb ist das MUL_MAT-Gate blind: nur Aufrufe, die `mmq_x=8` selektieren,
   lösen den Fehler aus, und die trifft `test-backend-ops` nicht.
2. **`OTHER=16` ist rückwirkend ebenfalls unsicher**, nicht nur langsam — der Q5_K-Sweep hat
   offenbar nie eine `mmq_x=8`-Kachel selektiert, der Bug lauerte dort aber genauso.
3. **Kein Fork-Bug** — die Formel stammt aus Upstream und fällt dort nicht auf, weil
   `256/warp_size` auf keiner unterstützten Karte `nwarps > mmq_x_min` ergibt.

**Bereits ausgeschlossen:**

- Akkumulator-Trunkierung (Fix `4e8441f69` — Crash bleibt)
- Blockgröße an sich: `OTHER=16` startet ebenfalls 1024 Threads und faultet **nicht**
- Knopf-Mismatch `Q8` ≠ `OTHER`: der q8_0-Pfad ist über
  `mmq_type_traits<…, GGML_TYPE_Q8_0>::nwarps` durchgängig konsistent parametrisiert

**Nächste Schritte:**

- [ ] `AMD_LOG_LEVEL=3` oder `rocgdb`, um den faultenden Kernel zu identifizieren —
      ohne diese Information ist weiteres Hypothesenbilden Zeitverschwendung
      (bisher drei Hypothesen, drei Fehlschläge)
- [ ] Shared-Memory-Berechnung gegen `smpbo` prüfen: greift der `mmq_x`-Filter in
      [../ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh#L4196) bei nwarps=16
      so, dass `mmq_x_best = 0` bleibt und der `switch` keinen Case trifft?
- [ ] `static_assert` bzw. harte Schranke gegen nwarps > 8, damit der Fall nicht still
      durchrutscht — unabhängig von der Ursache sinnvoll
- [ ] MUL_MAT-Gate um Shapes ergänzen, die den Fall abdecken (Gate ist hier blind)

- [x] **Q8_0-Modell vorhanden:** `/home/pat/data/models/Ornith-1.0-9B-Q8_0.gguf` (9,5 GB, nativ
      quantisiert, seit 2026-08-03 auf dem Server). Passt einzeln auf eine 32-GB-Karte →
      Single-GPU-Sweep ohne Multi-GPU-Rauschen. Gleiche Modellfamilie wie das
      `Ornith-1.0-9B-Q4_0`, mit dem der Q4_0-pp512-Rückstand in Phase 2.2 vermessen wurde.
- [ ] **Nicht** `q8_0-tinyllama-1.1b-chat-v0.3.gguf` als Messbasis verwenden — mit 1,1B zu klein,
      um `nwarps` aufzulösen (Phasen 3.2/4.3 landeten damit durchweg im Rauschen; bei pp512
      ~4760 t/s dominiert Launch-Overhead statt CU-Auslastung). Nur als schneller Smoketest.
- [ ] Sweep `-DGGML_MMQ_NWARPS_GFX906_Q8=4/8/16` via
      [../scripts/bench-gfx906-nwarps.sh](../scripts/bench-gfx906-nwarps.sh)
      (Rebuild je Konfiguration nötig — `nwarps` ist constexpr in `__launch_bounds__`)
- [ ] Je Konfiguration `test-backend-ops -o MUL_MAT` als Gate
- [ ] **Korrektheits-Caveat aus #23881 beachten:** bei nwarps=8 wurde ein
      "non-deterministic flicker" für `q5_1, m=16, n=1, k=32` berichtet
- [ ] Ergebnis in [gfx906-mmq-nwarps-tuning.md](gfx906-mmq-nwarps-tuning.md) und
      [../AGENTS.md](../AGENTS.md) nachtragen

**Aufwand:** ~1 Tag (plus Modell-Download). **Erwartung:** offen; Q8 hat andere
Registerlast als die K-Quants, die Occupancy-Klippe muss nicht an derselben Stelle liegen.

**Aufwand:** 1–2 Tage. **Gating-Funktion:** Falls hier deutliche Gewinne liegen, relativiert das den
Nutzen weiterer Kernel-Arbeit (Phase C).

---

## Phase C: Q4_0-pp512-Rückstand diagnostizieren

Der einzige **bekannte und quantifizierte** Rückstand (780 vs. 842 t/s). Tests A–D in Phase 2.2 haben
vec-load, nwarps, hipBLAS-Override und Flash Attention als Ursache bereits ausgeschlossen.
Arbeitsthese: Mainlines redesigntes MMQ-Subsystem (`ggml_cuda_mmq_get_config_*` pro Architektur) vs.
Legacy-Pfad im Fork.

- [ ] Diffs vom Server holen (Kommandos stehen in Phase 2.2 des Portierungsplans,
      `/opt/llama.cpp` vs. `/opt/mx-llama.cpp`: `mmq.cuh`, `mmvq.cu`, `ggml-cuda.cu`, `llama-graph.cpp`)
- [ ] These verifizieren: unterscheiden sich Tile-Größen (`mmq_y`) oder das Layout
      `block_q8_1_mmq` zwischen den Pfaden für Q4_0?
- [ ] Entscheidung dokumentieren zwischen den drei Optionen aus Phase 2.2:
      (1) Mainline-MMQ portieren, (2) Rückstand akzeptieren, (3) gezielt Legacy-Pfad tunen
- [ ] Bei Option 1: Risiko für fork-spezifische Features vorab prüfen
      (MMQ-Entscheidung aus MERGE_REPORT.md beachten — Fork-Tuning wurde bewusst behalten)

**Aufwand:** 2–3 Tage Diagnose, danach je nach Option 1 Tag bis 2 Wochen.

---

## Phase D: `sgemm.cuh` (FP32 GEMM) — Phase 3.1 des Portierungsplans

Einziger noch nicht portierter Kernel mit klarer Motivation: Fälle, in denen hipBLAS
`CUBLAS_STATUS_INTERNAL_ERROR` wirft.

- [ ] Erst prüfen, ob dieser Fehler auf der aktuellen ROCm-Version überhaupt noch auftritt —
      falls nein, ersatzlos streichen
- [ ] Nur bei bestätigtem Auftreten portieren und als Fallback-Pfad in
      `ggml_cuda_mul_mat_cublas` einhängen

**Aufwand:** 1 Woche. **Priorität:** niedrig — Korrektheits-Fallback, kein Performance-Hebel.

---

## Zurückgestellt: Phase 5 (q8-cache, graph-fusion, norm-fused-q8, gather-q8)

2–4 Wochen pro Baustein bei sehr hohem Risiko. Nach vier Phasen ohne messbaren Gewinn ist die
Erfolgswahrscheinlichkeit weiterer Kernel-Ports empirisch niedrig. **Nur angehen, wenn ein konkretes
VRAM-/Kontext-Problem gelöst werden soll** — nicht als Performance-Spekulation.

---

## Reihenfolge & Abhängigkeiten

```
Phase 0 (Hardware)  ──┬──> Phase A (nachmessen)
                      └──> Phase B (Sweeps) ──> Gating für Phase C/D
                                                Phase C (Q4_0-Diagnose)
                                                Phase D (sgemm, optional)
```

Phase 0 ist **blockierend**: ohne stabile Takte sind A, B und C nicht sinnvoll messbar.

---

## Querverweise

- [port-gfx906-kernels-from-skyne98.md](port-gfx906-kernels-from-skyne98.md) — Phasen 1–5, Status
- [gfx906-mmq-nwarps-tuning.md](gfx906-mmq-nwarps-tuning.md) — offener Q8_0-nwarps-TODO
- [../docs/gfx906/gfx906-optimization-notes.md](../docs/gfx906/gfx906-optimization-notes.md) — Teil B §10–12
  (Power/Clock/Fan-Tools), Teil C (MMQ-nwarps-Fund)
- [../AGENTS.md](../AGENTS.md) — Fork-Features, die bei Änderungen erhalten bleiben müssen
