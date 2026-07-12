# Plan: MMQ `nwarps`-Tuning für gfx906

**Ziel:** Prompt-Processing-Durchsatz (pp) auf gfx906 (MI50/MI60) erhöhen, indem die
MMQ-`nwarps`-Heuristik über die aktuelle Fork-Einstellung hinaus getunt wird.

**Grundlage:** [docs/gfx906-optimization-notes.md](../docs/gfx906-optimization-notes.md) Teil C
(llama.cpp Discussion [#23881](https://github.com/ggml-org/llama.cpp/discussions/23881)):
`nwarps` 4 → 16 brachte **+54 % pp** (Q8, MI60), **+76 %** auf MI50 32 GB, weil die
`256/warp_size`-Heuristik für non-MFMA-Karten unterdimensioniert ist.

**Status:** unabhängig vom Upstream-Merge (der Merge hat mmq.cuh nicht verändert). Reines
Performance-Experiment, keine Merge-Korrektur.

> **Umsetzungsstand: ABGESCHLOSSEN.** Abschnitt 3 (Code + CMake) implementiert,
> Sweep-Skript ([scripts/bench-gfx906-nwarps.sh](../scripts/bench-gfx906-nwarps.sh))
> gebaut, Sweep auf MI50 durchgeführt (Abschnitte 4–6). **Ergebnis: OTHER 4 → 8 = +23 %
> pp512** (qwen35 9B Q5_K, single GPU, MUL_MAT-Gate 2/2); 16 regressiert (Occupancy-Klippe).
> Neuer Default `GGML_MMQ_NWARPS_GFX906_OTHER = 8` in mmq.cuh, dokumentiert in
> docs/gfx906-optimization-notes.md Teil C. Q8-Knopf bleibt 8 (kein Q8-Modell zum Nachmessen).
> Zusätzlich zum Plan wurde `mmq_get_nwarps_compile_default()` mitgezogen (gfx906 → OTHER),
> weil load_tiles ohne explizites `nwarps`-Template davon abhängen — sonst Mismatch mit den
> Launch-Bounds bei angehobenem OTHER-Wert.
>
> **Gegenprobe Q6_K erledigt:** Qwen3.6 27B Q6_K zeigt dasselbe Muster (OTHER 4→8 = +19 %,
> 16 regressiert, Gate 2/2) → Default breit bestätigt.
> **Offene Nice-to-haves:** Q8_0-Modell besorgen und den 8→16-Fund aus #23881 für den
> Q8-Knopf verifizieren.

---

## 1. Ausgangslage (Ist-Zustand im Code)

Datei: [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh)

- **Host** `mmq_get_nwarps_host<type>(cc, warp_size)` (Z. 305–313): gfx906 (VEGA20) →
  `Q8_0 ? 512/warp_size (=8) : 256/warp_size (=4)`. Bestimmt Block-Dims + Shared-Mem-Größe
  (Z. 3998, 4111, 4155).
- **Device** `mmq_get_nwarps_compile<type>()` (Z. 321–332): gfx906 → `Q8_0 ? 8 : 4`. Fließt in
  `mmq_get_nwarps_device_type<type>()` → `__launch_bounds__` (Z. 3579–3585) und Kernel-Body.

Fazit Ist:
```
Q8_0            → nwarps 8   (bereits getunt, "improve pp with q8"-Commit)
alle anderen    → nwarps 4
```

## 2. Zentrale Randbedingung (bestimmt den Ansatz)

`nwarps` ist auf der Device-Seite **`constexpr`** (Template-Parameter + `__launch_bounds__`).
Es ist damit **compile-time-fixiert pro Quant-Typ**. Konsequenzen:

- **Keine Runtime-Env-Var möglich** ohne mehrere Kernel-Varianten zu kompilieren und zu
  dispatchen (zu teuer, verworfen).
- Host- und Device-nwarps **müssen exakt übereinstimmen** (sonst falsche Block-Dims /
  Shared-Mem-Layout → Fehlberechnung oder Launch-Fehler). Jede Änderung also **an beiden
  Stellen synchron**.
- A/B-Test daher über ein **Compile-Macro**, das beide Funktionen speist; pro Wert neu bauen.

## 3. Umsetzung

### 3.1 Überschreibbares Macro einführen
In `mmq.cuh` ein optionales Macro definieren, das die gfx906-Werte steuert, mit den heutigen
Werten als Default (verhaltensneutral, wenn nicht gesetzt):

```c
#ifndef GGML_MMQ_NWARPS_GFX906_Q8    // nwarps für Q8_0 auf gfx906
#define GGML_MMQ_NWARPS_GFX906_Q8    8
#endif
#ifndef GGML_MMQ_NWARPS_GFX906_OTHER // nwarps für alle anderen Typen auf gfx906
#define GGML_MMQ_NWARPS_GFX906_OTHER 4
#endif
```

### 3.2 Host- und Device-Heuristik auf das Macro umstellen
- `mmq_get_nwarps_host<type>` (gfx906-Zweig): `type == Q8_0 ? Q8 : OTHER`.
- `mmq_get_nwarps_compile<type>()` (`__gfx906__`-Zweig): dito, mit denselben Macros.
- Konsistenz-Assert/`static_assert` prüfen, dass Host == Device je Typ (wenn möglich).

### 3.3 Übergabe via CMake (für A/B ohne Code-Edit)
Optionaler Cache-Eintrag, z. B. in der HIP-Backend-CMakeLists, der die Macros an den
Compiler durchreicht:
```
-DGGML_MMQ_NWARPS_GFX906_Q8=<N> -DGGML_MMQ_NWARPS_GFX906_OTHER=<M>
```
So kann jeder Bench-Durchlauf mit einem reinen `cmake`-Flag konfiguriert werden.

> **Automatisierung:** Der komplette Sweep (configure → build → `test-backend-ops`-Gate →
> `llama-bench` → Ergebnistabelle) ist in [scripts/bench-gfx906-nwarps.sh](../scripts/bench-gfx906-nwarps.sh)
> umgesetzt. Baut in eine separate `build-nwarps-sweep/`-Dir (lässt `build/` unangetastet),
> nutzt ccache, mit den MI50-Standardflags (RCCL=ON, ROCWMMA_FATTN=OFF, LTO). Aufruf:
> `scripts/bench-gfx906-nwarps.sh ~/data/models/DEIN_MODELL.gguf`. Sweep/Bench über Env-Vars
> (`Q8_VALUES`, `OTHER_VALUES`, `BENCH_ARGS`, …) steuerbar.
>
> **Achtung TP-Load-OOM:** `-sm tensor` auf einem großen Modell OOMt beim Laden auf der
> 16-GB-Karte (Gerät 0 = 16368 MiB; even-split gibt ihr ~die Hälfte der Gewichte + KV +
> Compute-Buffer). **Das ist der zuvor gesehene `meta_buf`-Assert-Crash — kein Merge-Bug,
> sondern ein bekanntes Setup-Limit** (auch im Build-Skript notiert: „TP + MTP crasht mit
> ROCm OOM auf der 16GB-Karte"). Für den Sweep daher entweder `BENCH_ARGS="… -ts 1,2"`
> (Split Richtung 32-GB-Karte), single-GPU (`BENCH_ENV="HIP_VISIBLE_DEVICES=1"`) oder
> Layer-Split statt `-sm tensor`.

## 4. Benchmark-Matrix (auf echter MI50, `llama-bench`)

Pro Konfiguration `pp512, pp2048, pp8192` (tg zum Gegencheck, sollte ~gleich bleiben, da
compute-bound). Mindestens `-r 3`.

| Q8-Kandidaten | Nicht-Q8-Kandidaten |
|---|---|
| 8 (Baseline) | 4 (Baseline) |
| 12 | 8 |
| 16 | 12 |

- Getrennt pro **tatsächlich genutztem Quant-Typ** der Testmodelle (GGUF-`file_type`
  vorher prüfen — MMQ greift nur bei quantisierten Gewichten; FP16 = irrelevant).
- Modelle: mindestens ein dense + ein MoE (MoE nutzt den MMQ-id-Pfad); MTP-Modell separat.
- LDS/Occupancy-Klippe beachten: mehr nwarps ⇒ mehr Register/LDS-Druck; ab einem Punkt fällt
  pp wieder. Höchsten stabilen Wert nehmen, nicht blind 16.

## 5. Correctness-Gate (Pflicht vor jedem Übernehmen)

```
./build/bin/test-backend-ops -o MUL_MAT
```
Discussion meldete bei **nwarps=8** einen nicht-deterministischen Fehler bei
`q5_1, m=16, n=1, k=32`. Daher:
- Jeden gewählten nwarps-Wert **mehrfach** durch `test-backend-ops -o MUL_MAT` jagen.
- Speziell `q5_1` (und die anderen produktiv genutzten Typen) auf Flicker prüfen.
- Bei Instabilität: diesen Typ auf dem stabilen niedrigeren Wert belassen (Heuristik erlaubt
  per-Typ-Differenzierung).

## 6. Entscheidung & Rollout

1. Pro Quant-Typ den höchsten Wert wählen, der (a) messbar schneller und (b) in
   `test-backend-ops` stabil ist.
2. Werte als neue Defaults in die Macros eintragen.
3. **Separater Commit**, nicht in den Merge-Branch-Fixes vermischt — klar rückrollbar.
   Commit-Message mit den Bench-Zahlen (Modell, Quant, pp vorher/nachher, MI50).
4. In [docs/gfx906-optimization-notes.md](../docs/gfx906-optimization-notes.md) Teil C das
   Ergebnis (gemessene MI50-Werte) ergänzen.

## 7. Risiken / offene Punkte

- **Kein lokaler Bench möglich** (Assistent hat keine GPU) — alle Messungen laufen auf dem
  Server.
- MI50 (60 CUs) kann anders liegen als die Discussion-Zahlen (MI60 64 CUs / MI50 32 GB).
- Höhere nwarps können Shared-Mem-Bedarf über `smpbo` treiben → in `mmq.cuh:4155` wird
  `mmq_x` dann reduziert; Netto-Effekt muss gemessen, nicht angenommen werden.
- Wechselwirkung mit dem noch offenen **Tensor-Parallel-Load-Crash** (`ggml-backend-meta.cpp`
  Assert, MoE+MTP, ungleicher VRAM 16/32 GB) ist keine — aber die Bench-Modelle sollten erst
  laufen, wenn dieser Crash geklärt ist. Ggf. nwarps-Bench zunächst single-GPU / `-sm layer`.

## 8. Aufwand

- Code (Macro + zwei Heuristik-Zweige + CMake-Flag): klein, ~1 h.
- Benchmark-Sweep + Correctness-Gate: der Hauptaufwand, auf dem Server, iterativ.
