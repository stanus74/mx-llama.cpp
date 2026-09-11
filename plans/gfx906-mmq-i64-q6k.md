# Plan: `I=64` typspezifisch für Q6_K auf gfx906

**Angelegt:** 2026-09-11 · **Branch:** `gcn5` · **Status:** geplant, nicht gemessen

Letzter offener Tuning-Punkt der MMQ-Config. Auslöser ist
[PR #3 auf `mxxm-t/mx-llama.cpp`](https://github.com/mxxm-t/mx-llama.cpp/pull/3) von `mixa3607`
samt der Gegenmessung des Fork-Betreibers.

---

## 1. Ausgangslage im Code

`ggml/src/ggml-cuda/mmq-config-gcn5.cuh`, Zeilen 162–172, elf `CASE`-Zeilen für Q6_K:

```c
CASE(GGML_TYPE_Q6_K, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, fallback);
//                   ^^^     ^^^                                                  ^^^^
//              nthreads      I                                               stream_k
```

- `fallback=true`: J ∈ {8, 16, 32, 64}
- `fallback=false`: J ∈ {8, 16, 24, 32, 40, 48, 64}

Der Kommentar in derselben Datei (Z. 22–26) hält den bisherigen Sweep fest:

> `I` was swept over 64/96/128/160/192/256: 128 is optimal, **64 costs 34 % on Q5_K**, and every
> other value compiles but faults the GPU with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`

**Zwei Folgerungen daraus, die diesen Plan tragen:**

1. Der Sweep war **auf Q5_K** gemessen. **Q6_K bei `I=64` ist nie gemessen worden.** Das ist die Lücke.
2. `I=64` **läuft** bei `nthreads=512` — es war langsamer, nicht kaputt. Nur 96/160/192/256 faulten.
   Das Fault-Risiko dieses Versuchs ist damit klein, anders als bei einem freien `I`-Sweep.

---

## 2. Evidenzlage

| Messung | `mixa3607` | `mxxm-t` (Gegenmessung) | **eigene** |
|---|---|---|---|
| Q6_K `I=64`, Kernelzeit | −35,0 % | −37,1 % dense / −46,1 % MoE | **nicht gemessen** |
| Q5_K `I=64` | +19,5 % pp | −9,6 % Kernel / −5,0 % pp | **−34 %** |
| Q4_K `I=64` | ~−53 % | — | — |
| bit-exakt | ja | ja für Q6_K, **nein** für Q5_K | irrelevant, s. u. |

**Q6_K:** zwei unabhängige Quellen, gleiche Richtung, ähnliche Größenordnung. Belastbarste
Fremdzahl, die dieses Projekt bisher gesehen hat.

**Q5_K:** `mxxm-t` und ich stimmen in der Richtung überein, `mixa3607` steht allein. Ursache ist im
Thread benannt — dessen Gewinn hängt am Compilerflag `-mllvm -amdgpu-sched-strategy=max-ilp`
(`mxxm-t`: *„These cost −9.6 % on current base build … but gain +3.1 % under max-ilp"*).
**Q5_K wird in diesem Plan nicht angefasst.**

**Bit-Exaktheit ist hier kein Kriterium.** Unsere Q6_K-Config fährt `stream_k=true`, und `stream_k`
zerlegt die K-Dimension — die Summationsreihenfolge weicht ohnehin ab. Das ist in
[mainline-fork-gcn5-patch.md](mainline-fork-gcn5-patch.md), Abschnitt „Schritt 3.4 im Detail",
bereits abgehandelt.

**Der Mechanismus im PR ist widerlegt.** `mixa3607` begründet mit „cutting VGPR/scratch";
`mxxm-t` hat nachgesehen: *„85 VGPRs against 84 with ScratchSize 0 in both"* — es spillt nichts,
nur die Occupancy wandert 2 → 3. Messung behalten, Erklärung verwerfen.

### Warum das kein Copy-Paste ist

| Parameter | `mixa3607` | hier |
|---|---|---|
| `nthreads` | 256 | **512** |
| `I` | 64 | **128** |
| `stream_k` | false | **true** |
| `sram_layout` | `Q6_K` | `Q6_K` ✓ |
| `occupancy` | 2 | 2 ✓ |

Dreifach konfundiert. Dass `nthreads` dabei nicht nebensächlich ist, zeigt `mxxm-t` für Q5_K:
*„512 at the inherited I=128 gives **956.5 vs 610.9** for I=64 here"* — unsere Kombination gewinnt
dort um 56 %. Für Q6_K ist die Frage offen, **deshalb wird nur `I` bewegt.**

---

## 3. Randbedingungen

**Aus [AGENTS.md](../AGENTS.md), Abschnitt „gfx906-Kernel-Tuning" — die wichtigste:**

> **Achtung:** `test-backend-ops -o MUL_MAT` läuft auch bei 16 sauber durch (2/2) — das Gate deckt
> die Shapes echter Modelle nicht ab und kann diesen Fehler nicht finden. Jede nwarps-Änderung
> zusätzlich mit `llama-bench` auf einem echten Modell verifizieren, nicht nur mit dem Gate.

Der Präzedenzfall: `nwarps=16` passierte das Gate und **crashte** die GPU auf echten Modellen
(`Memory access fault … Write access to a read-only page`). `test-backend-ops` ist hier also
**notwendig, aber nicht hinreichend**. Das gilt für `I` genauso wie für `nwarps`.

Weiter:

- **Keine Builds/Tests ungefragt starten** (AGENTS.md, „Build & Test"). Dieser Plan wird nicht
  ausgeführt, bis der Betreiber ihn freigibt.
- **Build läuft auf dem Server** `pat@192.168.178.71` (`x99`), Worktree `/opt/mx-llama-gcn5`.
  `ssh` steht bewusst nicht in der Allowlist — Freigabe pro Aufruf nötig, oder der Betreiber führt
  die Schritte selbst aus.
- **`HSA_XNACK=0` muss gesetzt sein.** Seit dem Fix vom 2026-09-06 liegt das in
  `~/.config/gpu-env.sh` und wird auch über nicht-interaktives SSH geladen. Kontrolle trotzdem:
  `ssh x99 'env | grep ^HSA'`, und im Bench-Log muss `gfx906:sramecc+:xnack-` stehen. Bei `xnack+`
  stürzt MoE-Prefill ab.
- **VRAM vorher prüfen.** llama-swap hält Modelle geladen; ein Benchmark gegen belegten Speicher
  scheitert mit `GGML_ASSERT(meta_buf_ctx->bufs[i])` — ein verkleidetes OOM, das wie eine
  Regression aussieht. Also `pgrep llama-server` und `rocm-smi --showmeminfo vram`.
  **Den Produktions-llama-swap nicht abschießen.**
- **Nicht auf `master` committen.** Arbeit auf `gcn5`, Commit/Push nur auf Aufruf.

---

## 4. Umsetzung

### Schritt 4.0 — zuerst klären, welche `CASE`-Zeile überhaupt feuert

**Bevor irgendetwas geändert wird.** `J` ist kein freier Parameter, sondern der Suchschlüssel des
`CASE`-Makros ([mmq.cuh:206](../ggml/src/ggml-cuda/mmq.cuh#L206)); der tatsächliche Wert ergibt sich
aus der Batchgröße. Wer die falschen Zeilen ändert, misst nichts und hält das für „kein Effekt".

Also: bei pp4096 feststellen, welches `J` und welcher `fallback`-Zweig gezogen wird — über ein
temporäres `printf` im Dispatch oder durch Rückrechnung aus der Batchgröße. Ergebnis hier eintragen:

```
pp4096 zieht:  J = ____   fallback = ____
tg128  zieht:  J = ____   fallback = ____
```

Dieser Schritt ist nicht optional. Der Repack-Fund vom 2026-09-05 war genau dieser Fehler in der
anderen Richtung — dokumentierter Code, der nie ausgeführt wurde.

### Schritt 4.1 — Variante A (empfohlen, konservativ)

Nur die `J=64`-Zeilen auf `I=64`, alles andere unangetastet. Das spiegelt den PR (`mixa3607`:
*„J=64 only"*) und trifft gezielt den Prefill, wo der Gewinn behauptet wird. Decode war in beiden
Fremdmessungen unverändert, bleibt hier also per Konstruktion unberührt.

```c
// Zeile 165 und 172: I 128 -> 64
CASE(GGML_TYPE_Q6_K, 512, 2, 64, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
CASE(GGML_TYPE_Q6_K, 512, 2, 64, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
```

Falls Schritt 4.0 zeigt, dass pp4096 ein anderes `J` zieht, gilt stattdessen **jenes** `J`.

### Schritt 4.2 — Variante B (nur falls A trägt)

Alle elf Q6_K-Zeilen auf `I=64`. Erweitert die Wirkung auf kleine Batches, also auch auf Decode.
Erst nach A messen, damit die Wirkung zuordenbar bleibt.

---

## 5. Messprotokoll

**Baseline zwingend neu messen.** Die Zahlen unten stammen von `b10288`, HEAD ist `b10873+`, und
der Modellbestand hat sich zwischenzeitlich geändert (`Ornith-1.0-35B` → `Ornith-1.5-35B-A3B`).
Ein Vergleich über Buildstände und Modellversionen hinweg ist wertlos.

| Modell | Typ | letzte bekannte pp4096 | Stand |
|---|---|---:|---|
| `Ornith-1.5-35B-A3B-FULLY-OBLITERATED.Q6_K` | MoE | 1115,42 ± 2,66 (auf 1.0) | b10288 |
| `Qwen3.8-27B-UD-Q6_K_XL` | dense | 383,73 ± 0,74 | 2026-09-06 |

Beide Modelle, weil die Fremdmessung dense und MoE unterschiedlich stark trifft
(−37,1 % gegen −46,1 %).

```bash
# auf x99, im Worktree /opt/mx-llama-gcn5
./scripts/mx-compile.sh                      # inkrementell, ~2 Min mit ccache

HIP_VISIBLE_DEVICES=0,1 build/bin/llama-bench \
  -m ~/data/models/Ornith-1.5-35B-A3B-FULLY-OBLITERATED.Q6_K.gguf \
  -ngl 99 -fa 1 -sm tensor -lm dio -r 5 -p 4096 -n 128
```

`-r 5`, nicht `-r 3`. Der SDMA-Fehlalarm vom 2026-09-06 zeigte bei `-r 3` einen scheinbaren
−24-%-Effekt, der bei `-r 5` verschwand.

**Reihenfolge, verbindlich:**

1. **Kontrollmessung:** Baseline-Build **zweimal** laufen lassen. Ohne das ist Nichtdeterminismus
   nicht von Wirkung zu trennen. Auf dieser Maschine liegt die tg-Streuung selbst bei `-r 5` bei
   ±4 t/s — Decode-Aussagen unter ~15 % sind nicht belastbar.
2. Variante A bauen, `test-backend-ops -o MUL_MAT` über `--check` (**notwendig, nicht hinreichend**).
3. `llama-bench` auf **beiden** Q6_K-Modellen — das ist das eigentliche Gate (AGENTS.md).
4. Ausgabevergleich: `llama-cli` mit festem Seed, Greedy, gegen den Baseline-Build.
   **`-no-cnv -st </dev/null` plus `head -c`-Deckel** — llama-cli ist seit b10240 eine Chat-UI und
   schreibt über SSH sonst GB-weise `> `-Prompts.
5. Erst wenn 1–4 tragen: Variante B, gleiches Protokoll.

---

## 6. Erfolgs- und Abbruchkriterien

**Übernehmen**, wenn auf **beiden** Q6_K-Modellen pp4096 reproduzierbar steigt, die Streuungen sich
nicht überlappen, `test-backend-ops` fehlerfrei ist und der Ausgabevergleich plausibel bleibt.

**Verwerfen**, wenn eines zutrifft:

- pp fällt oder stagniert auf einem der beiden Modelle
- `test-backend-ops` meldet Fehlschläge
- GPU-Fault oder `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
- die Differenz liegt innerhalb der Streuung der Kontrollmessung

**Erwartung: deutlich weniger als die zitierten 35 %.** Deren Gewinn stammt teilweise aus
`nthreads=256`; hier stehen schon 512 mit `I=128`, also die laut `mxxm-t` für Q5_K bessere
Kombination. Fremde Prozentzahlen sind in diesem Projekt **viermal in Folge geschrumpft** (BF16
wirkungslos, q8_1 auf ⅓, Lane dispatch hardwareabhängig, Repack toter Code). Ein einstelliger
pp-Gewinn wäre ein gutes Ergebnis, null ein plausibles.

---

## 7. Risiken

| Risiko | Einschätzung |
|---|---|
| GPU-Fault bei `I=64` | **klein** — `I=64` lief im bisherigen Sweep, es war nur auf Q5_K langsamer |
| Falsche `CASE`-Zeile geändert → „kein Effekt" | durch Schritt 4.0 adressiert |
| Regression bei Decode | Variante A rührt Decode nicht an; B muss dafür separat gemessen werden |
| Q4_K versehentlich mitgeändert | Q4_K verliert bei `I=64` ~53 % — **Zeilen 138–148 nicht anfassen** |
| Messung gegen belegten VRAM | durch VRAM-Prüfung vor dem Lauf adressiert |
| Rebase-Konflikt später | keiner: die Datei ist fork-exklusiv, Upstream berührt sie nicht |

---

## 8. Aufwand

Codeänderung zwei Zeilen. Build ~2 Min inkrementell. Vier Bench-Läufe à ~5 Min plus
Kontrollmessung. **Insgesamt unter einer Stunde**, sofern der VRAM frei ist.

---

## 9. Ergebnis: ❌ verworfen (2026-09-11)

**`I=64` für Q6_K ist auf dieser Maschine deutlich schlechter, nicht besser.**

Gemessen auf `x99`, Build `1cb6d4823` (**b10907**), 2 GPUs `-sm tensor -lm dio -fa 1 -r 5`,
`gfx906:sramecc+:xnack-` bestätigt.

### Schritt 4.0 — welche `CASE`-Zeile feuert

`ggml_cuda_mmq_get_J_max` ([mmq.cuh:372](../ggml/src/ggml-cuda/mmq.cuh#L372)) startet bei
`min(ne11, 512)` und läuft in 8er-Schritten **abwärts**, bis ein `CASE` trifft. Die Q6_K-Zeilen
enden bei J=64, also:

| Betriebspunkt | J | Folge |
|---|---|---|
| pp4096 | **64** | Variante A trifft genau hier |
| tg / MTP-Draft (Batch < 8) | **0** | `J_max` liefert 0 → MMQ läuft gar nicht |

`fallback = ne01 % 128 != 0` ([mmq.cu:129](../ggml/src/ggml-cuda/mmq.cu#L129)), hängt an der
Zeilenzahl des Gewichts — beide Zweige können feuern, deshalb wurden beide J=64-Zeilen geändert.

**Damit war vorab klar: Variante A kann Decode konstruktiv nicht beeinflussen.** Die Messung
bestätigt das (tg über alle Läufe unverändert).

### Kontrollmessung — das Rauschen ist winzig

| Lauf | pp4096 |
|---|---:|
| Baseline 1 | 1192,31 ± 1,87 |
| Baseline 2 | 1192,26 ± 3,11 |
| Baseline nach Rückstellung | 1194,19 ± 2,85 |

Streuung über drei Läufe **0,16 %**. Jede Differenz über ~0,5 % ist damit echt — und die
Rückstellung ist belegt, das Ergebnis unten also kein Artefakt.

### Messung

| Modell | Baseline `I=128` | `I=64` | Δ |
|---|---:|---:|---:|
| `Ornith-1.5-35B-A3B` Q6_K (MoE, 27,19 GiB) | 1192,3 | **1018,37 ± 2,45** | **−14,6 %** |
| `Qwen3.8-27B-Cold-Fusion` Q6_K (dense, 21,95 GiB) | 388,44 ± 1,12 | **293,47 ± 0,39** | **−24,5 %** |

tg128 unverändert: 59,3 → 59,0 (MoE), 24,29 → 24,35 (dense).

Correctness-Gate war dabei **sauber** — q4_K 31 OK, q5_K 25 OK, q6_K 12 OK, q8_0 19 OK,
`MUL_MAT_ID` q6_K 2 OK, **0 FAIL**, kein GPU-Fault. Genau der von `AGENTS.md` beschriebene Fall:
**das Gate hätte diese Regression nicht gefunden.** Nur `llama-bench` auf echten Modellen hat sie
gezeigt.

### Was von der Fremdzahl übrig bleibt

Behauptet waren −35,0 % Kernelzeit (`mixa3607`) und −37,1 % dense / −46,1 % MoE (`mxxm-t`s
Gegenmessung). Gemessen wurde hier das **Gegenteil**, und zwar auf beiden Modellklassen.

Die Erklärung liegt nach Aktenlage bei `nthreads`. `mixa3607` fährt `nthreads=256` **mit** `I=64`;
diese Config fährt `512` mit `I=128`. `mxxm-t` hatte für Q5_K genau darauf hingewiesen —
*„512 at the inherited I=128 gives 956.5 vs 610.9 for I=64 here"* —, und dieselbe Kopplung gilt
offenbar für Q6_K: `I=64` ist bei **256** Threads gut und bei **512** schlecht. `I` und `nthreads`
sind nicht unabhängig, weil beide die Arbeit pro Thread bestimmen.

**Der Kommentar in `mmq-config-gcn5.cuh` („128 is optimal") gilt damit auch für Q6_K** — er war
nur auf Q5_K belegt und ist jetzt auf Q6_K nachgemessen.

### Nicht ausgeführt

**Variante B entfällt.** Sie erweitert dieselbe Änderung auf weitere J-Werte; nachdem A schon
regressiert, könnte sie nur zusätzlich schaden.

### Offen geblieben

Ob `nthreads=256 + I=64` (die **vollständige** Fremdconfig) die zitierten Zahlen reproduziert.
Das würde die Kopplung beweisen statt sie nur plausibel zu machen — ist aber ein anderer Versuch
als dieser und würde `nthreads` unter den bisherigen Optimalwert senken.

### Kein Nebenbefund: die Asymmetrie war längst dokumentiert

`ggml_cuda_init` meldet **16 368 MiB** (Device 0, „AMD Instinct MI50/MI60") und **32 752 MiB**
(Device 1, „AMD Radeon Graphics"). Das hatte ich zunächst als neuen Befund notiert — falsch, es
steht bereits in [multi-gpu-split-modes.md](../docs/gfx906/multi-gpu-split-modes.md) („MI50 16 GB +
MI50/Radeon-VII 32 GB") und in [anleitung-opti-gfx906.md](../docs/gfx906/anleitung-opti-gfx906.md)
mit Tabelle, OOM-Warnung für die 16-GB-Karte und der MTP-Crash-Analyse. **Kein Handlungsbedarf.**

Lehre fürs nächste Mal: vor „Nebenbefund" erst die eigene Doku greppen.
