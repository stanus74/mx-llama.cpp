# Plan: Fork auf Mainline reduzieren (Neuaufbau mit belegten Patches)

**Erstellt:** 2026-08-03 · **Status:** Vorschlag, noch nicht begonnen
**Ausgangsbranch:** `merge-upstream-20260803` (enthält `b10238`)

---

## Motivation

Der Fork weicht in **74 Dateien** um ~12.400 eingefügte / ~6.700 gelöschte Zeilen von
`b10238` ab. Dem steht gegenüber, was diese Divergenz nach den Messungen aus den
Phasen 1–4 und vom 2026-08-03 tatsächlich einbringt:

| Bestandteil | Umfang | Gemessener Nutzen | Quelle |
|---|---:|---|---|
| MMQ-Tuning | **~15 Zeilen** | **+7 % pp** *(gegen Mainline; +19…28 % galten nur gegen `nwarps=4` im Legacy-Pfad)* | Phase G, 2026-08-03 |
| Multi-Stage TP (`-tps`) | ~885 + 728 Z. | **≈ +3 % pp** über upstreams `-sm tensor` | Phase G, 2026-08-03 |
| MTP-Optimierungsschicht | verteilt | **+19 % Prefill** | MERGE_REPORT, Rauchtest 2026-08-03 |
| GCN-Repack (`repack-gcn.cu`) | 2283 Z. | **nie isoliert gemessen** | — |
| skyne98-Kernel-Ports | ~3000 Z. | **0** (MMVQ langsamer, MMF/FATTN-Q8 im Rauschen) | `port-gfx906-kernels-from-skyne98.md` |
| Legacy-MMQ in `mmq.cuh` | 4465 Z. Diff | **negativ:** Q4_0 pp512 −7 % vs. Mainline | Phase 2.2, Tests A–D |

**Kernbefund:** Der größte Gewinn steckt in ~15 Zeilen, die größte Divergenz enthält
gleichzeitig den einzigen bekannten Rückstand, und ~3000 Zeilen portierter Kernel liefern
nachweislich nichts.

**Eigentliches Ziel ist nicht Performance, sondern Wartbarkeit:** Der Merge vom 2026-08-03 hat
vier Dateien der empfindlichsten Fork-Komponente (MTP) **ohne Konfliktmarker** angefasst. Das
ging gut aus, war aber Glück, nicht Struktur. Wenige benannte Patches auf Mainline machen
Upstream-Syncs zu Routine statt zu Risiko.

---

## Phase G: Gate — TP-Vergleich (blockierend, vor allem anderen)

Upstream hat seit #19378 **eigenes backend-agnostisches Tensor-Parallel** (`-sm tensor`,
experimentell), weiterentwickelt in #22129, #22299, #25028. Die +32 % des Forks stammen aus
einem Vergleich gegen Mainline — **unklar ist, ob die Mainline-Seite dabei `-sm tensor` nutzte
oder ohne TP lief.**

Davon hängt der teuerste Teil des ganzen Plans ab: Custom-AllReduce + Meta-Device sind
~1600 Zeilen. Liefert upstreams TP den Großteil des Gewinns bereits, entfallen sie ersatzlos.

### ✅ ERLEDIGT 2026-08-03 — Ergebnis: Multi-Stage-TP **nicht** portieren

Modell `Ornith-1.0-9B-Q8_0` (qwen35 9B Q8_0), `llama-bench -ngl 99 -fa 1 -r 3 -p 2048 -n 128`.
Mainline `/opt/llama.cpp` auf `f2b52a87e` (= `b10238` + 1), Fork auf `aacf2aeb5`.
**Vier** statt drei Konfigurationen — die vierte war nötig, weil Konfiguration C sonst zwei
Fork-Vorteile gleichzeitig enthält (TP **und** MMQ-Tuning):

| | pp2048 | tg128 |
|---|---:|---:|
| **A** Mainline, 1 GPU | 685,53 ± 0,55 | 49,31 ± 0,16 |
| **D** Fork, 1 GPU | 733,90 ± 0,43 | 50,88 ± 0,07 |
| **B** Mainline, `-sm tensor` | 1114,53 ± 1,15 | 59,63 ± 4,13 |
| **C** Fork, `-sm tensor -tps 2` | 1231,67 ± 0,54 | 59,49 ± 4,06 |

**Zerlegung (pp2048):**

- **Upstream-TP allein: +62,6 %** (A→B) — der ganz überwiegende Teil des Multi-GPU-Gewinns
- **MMQ-Tuning des Forks: +7,1 %** (A→D), sauber isoliert auf einer GPU
- **Fork-TP über Upstream-TP: +10,5 % roh** (B→C), darin steckt das MMQ-Tuning erneut.
  Herausgerechnet (1231,67 / (1114,53 × 1,071)) bleiben für Multi-Stage-TP selbst **≈ +3 %**.

**Konsequenz:** ~1600 Zeilen Custom-AllReduce und Meta-Device — ausgerechnet die Komponenten,
die bei Upstream-Merges die meiste Unruhe stiften — liefern gegenüber upstreams eigenem TP
etwa **drei Prozent**. Klarster Streichkandidat des Plans. → **Phase 2 entfällt.**

**Die früher berichteten +32 % sind damit erklärt:** der Vergleich lief gegen Mainline *ohne*
TP. Gegen Mainline *mit* `-sm tensor` bleibt fast nichts übrig.

**Nebenbefunde:**

- **Das MMQ-Tuning bringt gegenüber Mainline +7 %, nicht +28 %.** Die +28 % waren `nwarps`
  4→8 *innerhalb des Legacy-Pfads*; Mainlines RDNA2-Fallback-Config ist deutlich besser als
  `nwarps=4`. Phase 1 bleibt lohnend, aber die Erwartung ist entsprechend zu korrigieren.
- **tg ist bei TP nicht unterscheidbar** (59,63 vs. 59,49 bei σ ≈ 4). Token-Generierung ist
  bandbreiten-, nicht parallelisierungslimitiert.

**Belastbarkeit:** σ zwischen 0,43 und 1,15 auf allen pp-Messungen — die bekannten
Taktschwankungen haben hier nicht gestört, Phase 0 war für diesen Vergleich nicht blockierend.
Einschränkung: **ein** Modell, **eine** Quantisierung, nur pp2048/tg128. Vor einer endgültigen
Löschung des TP-Codes mit einem zweiten Modell gegenprüfen.

---

## Phase 1: MMQ-Config für GCN5 — **neu implementieren, nicht portieren**

Der beste Nutzen/Aufwand-Quotient im gesamten Fork. Es ist **kein Fork-Feature**, sondern die
Korrektur einer für GCN5 falsch gewählten Konfiguration.

### Befund: auf Mainline sieht dieser Patch anders — und besser — aus

Mainline hat MMQ umgebaut. `nwarps` ist dort **kein Define mehr**, sondern folgt aus
`config.nthreads / warp_size` ([b10238 `mmq.cuh:1396`]), und die Config kommt aus
`ggml_cuda_mmq_get_config_*` **pro Architektur**. Die Auswahl in `b10238`:

```c
if (GGML_CUDA_CC_IS_AMD(cc)) {
    if (GGML_CUDA_CC_IS_CDNA(cc))    return ..._cdna(...);
    if (GGML_CUDA_CC_IS_RDNA4(cc))   return ..._rdna4(...);
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) return ..._rdna3_5(...);
    if (GGML_CUDA_CC_IS_RDNA3(cc))   return ..._rdna3(...);
    return ..._rdna2(...);           // <-- gfx906 landet hier
}
```

**gfx906 (GCN5) hat keinen eigenen Fall und bekommt die RDNA2-Config** — auf einer Karte mit
doppelter Wave-Größe (64 statt 32). Das ist exakt dieselbe Unterdimensionierung wie die vom
Fork gefundene `256/warp_size`-Heuristik, nur in der neuen Struktur.

**Konsequenz:** Der Patch wird auf Mainline **kleiner und sauberer** — ein neuer
`ggml_cuda_mmq_get_config_gcn5`-Zweig statt Define-Overrides plus 4465 Zeilen Legacy-MMQ.
Genau die Form, die upstream vorgesehen hat, und damit als PR realistisch.

### Aufgaben

- [ ] `ggml_cuda_mmq_get_config_gcn5` (bzw. `_vega20`) anlegen und im Dispatch vor dem
      RDNA2-Fallback einhängen
- [ ] Ausgangswert: `nthreads = 8 * 64 = 512` (entspricht dem gemessenen `nwarps = 8`)
- [ ] Gegen **Mainline-Baseline** messen, nicht gegen den Fork — die Zahlen aus
      `gfx906-naechste-optimierungsschritte.md` B.1 gelten für den Legacy-Pfad
- [ ] Prüfen, ob der Q4_0-pp512-Rückstand (−7 %) damit ohnehin verschwindet
- [ ] Upstream-PR erwägen — bei Aufnahme **null Pflegeaufwand**

**Nicht mitnehmen:** Der Akkumulator-Off-by-one (`4e8441f69`) und die `static_assert`-Schranke
(`aacf2aeb5`) betreffen den **Legacy**-Pfad. Ob Mainlines redesigntes MMQ dieselbe
Trunkierung hat, ist offen — vor einer Upstream-Meldung erst dort nachrechnen.

**Hinweis:** Upstream hat eine Anti-AI-PR-Policy. Der Patch ist klein und gut messbar; die
Herleitung sollte als eigene Messung dargestellt werden, nicht als Werkzeugausgabe.

---

## Phase 2: TP — ❌ ENTFÄLLT (Phase G, 2026-08-03)

Multi-Stage-TP liefert gegenüber upstreams `-sm tensor` nur ≈ +3 % pp2048. Der Aufwand
(~1600 Zeilen in `tp-allreduce.cu` und `ggml-backend-meta.cpp`, plus das `-tps`-Flag) steht
in keinem Verhältnis dazu — zumal genau diese Dateien den Merge-Aufwand treiben.

- [x] Entscheidung: **nicht portieren**, upstreams `-sm tensor` genügt
- [ ] Vor dem endgültigen Verwerfen: Gegenprobe mit einem zweiten Modell (Phase G lief auf
      einem einzigen 9B-Q8_0)
- [ ] `-tps` aus der Doku entfernen bzw. auf `-sm tensor` umleiten (AGENTS.md, README-Hinweise)

---

## Phase 3: GCN-Repack — ⚠️ ERGEBNISLOS 2026-08-03 (Pfad wird nicht betreten)

Messung `ornith-1.0-9b-Q5_K_M`, single GPU (32-GB-Karte), `-r 3 -p 2048 -n 128`:

| Konfiguration | pp2048 | tg128 |
|---|---:|---:|
| Fork, Repack **an** (Default) | 679,69 ± 0,80 | 48,86 ± 0,14 |
| Fork, Repack **aus** (`GGML_CUDA_REPACK=0`) | 679,54 ± 0,69 | 48,97 ± 0,09 |
| Mainline (kein Repack, kein MMQ-Tuning) | 580,44 ± 0,73 | 46,53 ± 0,17 |

**Kein Unterschied zwischen an und aus — aber das ist keine Aussage über den Nutzen von
Repack, sondern über seine Aktivierung.** Positivkontrolle mit `GGML_CUDA_REPACK_Q8_0=1` auf
`Ornith-1.0-9B-Q8_0` (laut Code-Kommentar „+43 % Prefill auf einem reinen Q8_0-Modell"):

| | pp2048 | tg128 |
|---|---:|---:|
| Q8-Repack aus (Default) | 733,81 ± 0,56 | 50,97 |
| Q8-Repack an | 733,85 ± 0,48 | 50,98 |

**Zwei unabhängige Schalter ohne jede Wirkung → der Repack-Pfad wird nie betreten.**
Phase 3 ist damit **ergebnislos, nicht negativ**. Ohne die Positivkontrolle wäre hier
fälschlich „2283 Zeilen ohne Nutzen, streichen" gelandet.

**Codeinspektion — mutmaßliche Ursache:** In [../src/llama-model.cpp](../src/llama-model.cpp#L1063)
fügt `make_gpu_buft_list` erst den **Default**-Buffer-Type ein (Z. 1063) und danach die
Extra-Bufts inkl. Repack (Z. 1074). Wird die Liste nach dem ersten passenden Eintrag
durchsucht, gewinnt immer der Default. **Keine Merge-Regression** — die Reihenfolge ist in
`port-skyne98-gfx906` und nach dem `b10238`-Merge identisch, der Zustand bestand also vorher.

**Wertvoller Nebenbefund:** Der Fork schlägt Mainline bei Q5_K um **+17,1 %** (580 → 679) —
deutlich mehr als die +7 % bei Q8_0. Da Repack nachweislich inaktiv ist, stammt dieser
Vorsprung **vollständig aus dem MMQ-Tuning**. Das stärkt Phase 1: bei K-Quants ist der
MMQ-Config-Patch der größere Hebel.

### Nächster Schritt: Instrumentierung statt weiterer Benchmarks

`llama-bench -v` protokolliert die Buffer-Type-Zuordnung nicht, weitere Messläufe bringen
daher nichts. Stattdessen:

- [ ] Log-Zeile in `ggml_backend_cuda_repack_buffer_type()`
      ([../ggml/src/ggml-cuda/repack-gcn.cu](../ggml/src/ggml-cuda/repack-gcn.cu#L2252)) —
      wird der Typ überhaupt erzeugt?
- [ ] Log-Zeile im Repack-Upload-Pfad — wird je ein Tensor tatsächlich repackt?
- [ ] Falls „erzeugt, aber nie benutzt": Auswahllogik in `select_weight_buft` /
      `make_gpu_buft_list` prüfen; testweise Repack **vor** dem Default einreihen
- [ ] Erst mit aktivem Pfad die eigentliche A/B-Messung wiederholen

**Entscheidung vertagt.** Solange nicht feststeht, ob Repack je aktiv war, ist weder
„übernehmen" noch „streichen" begründbar. Sollte sich zeigen, dass der Pfad **nie** benutzt
wurde, wären die 2283 Zeilen toter Code — dann ist die Streichung trivial begründet.

---

## Phase 4: Fallenlassen

- [ ] **MTP-Optimierungsschicht.** +19 % Prefill gegen die höchste Wartungslast im Fork; sie
      hängt an Pre-Norm-Semantik, die upstream in jedem Zyklus anfasst (allein in `b10238`
      vier MTP-Commits). Upstream-MTP-Basis bleibt ja erhalten — es entfällt nur die
      Fork-Optimierung darauf.
- [ ] **skyne98-Kernel-Ports** (RoPE, vecdotq, MMVQ, MMF, FATTN-Q8): messbar ohne Gewinn.
- [ ] **Legacy-MMQ-Divergenz:** mit dem Wechsel auf Mainlines redesigntes MMQ-Subsystem
      verschwindet der Q4_0-pp512-Rückstand von −7 % automatisch. Danach prüfen, ob der
      `nwarps`-Patch dort überhaupt noch nötig/anwendbar ist — Mainline hat
      `ggml_cuda_mmq_get_config_*` pro Architektur.

---

## Zielzustand

```
b10238 (oder neuer)
  └─ patch/mmq-nwarps-gfx906        (~15 Zeilen, ggf. upstream)
  └─ patch/tp-multistage            (nur falls Phase G es belegt)
  └─ patch/repack-gcn               (nur falls Phase 3 es belegt)
```

Statt 74 divergierender Dateien: zwei bis drei benannte Patches mit je einer Messung als
Rechtfertigung. Upstream-Sync wird zum Rebase dieser Patches.

---

## Betriebsmodell: vom mergenden zum rebasenden Fork

Das ist der eigentliche Hebel — wichtiger als jede einzelne Patch-Entscheidung.

**Heute (mergend):** Upstream wird hereingemerged. Die Historie wächst, der eigene Anteil
verschwimmt über 74 Dateien, und Git wendet upstream-weite Änderungen ohne Marker an. Genau
so konnte der `b10238`-Merge vier Dateien der MTP-Komponente anfassen, ohne einen einzigen
Konflikt zu melden.

**Ziel (rebasend, Patch-Queue):**

```bash
git checkout -b gfx906 b10238        # Basis = getaggter Mainline-Stand
# darauf N saubere, thematisch getrennte Commits
```

Beim nächsten Release:

```bash
git fetch upstream --tags
git rebase --onto b10xyz b10238 gfx906
```

**Was das praktisch ändert:**

- `git log b10238..gfx906` zeigt **immer** exakt die eigenen Patches — der Fork-Anteil ist
  jederzeit vollständig überblickbar.
- Konflikte treten **pro Patch** auf statt diffus über den Baum.
- Ein Patch, der sich nicht mehr sauber aufsetzen lässt, ist ein **lautes Signal** — nicht
  ein stiller Auto-Merge. Das ist die direkte Antwort auf die wichtigste Lektion in AGENTS.md.

### Verifikationskette pro Patch (verbindlich)

Die Reihenfolge ist nicht beliebig — genau diese Kette hätte am 2026-08-03 zwei
Fehlschlüsse verhindert:

1. `test-backend-ops -o MUL_MAT` — notwendig, aber **nicht hinreichend**: das Gate hat den
   `nwarps=16`-Fault mit 2/2 durchgewinkt.
2. `llama-bench` auf einem **echten Modell** — deckt die Shapes ab, die das Gate nicht trifft.
3. Inferenz-Rauchtest (`llama-cli`, ggf. mit `--spec-type draft-mtp`) — deckt Semantikfehler
   ab, die weder Gate noch Benchmark sehen.

### Kadenz

Nicht jedem Tag folgen. Ein Release alle zwei bis vier Wochen rebasen genügt; der vorhandene
self-hosted Workflow `build-self-hosted` kann den Rebase-Build automatisch prüfen.

---

## Form: kein frischer Klon nötig

Ein neues Repository ist **nicht** erforderlich und wäre ein Nachteil — die Historie des alten
Forks dokumentiert, was wie gemessen wurde. Stattdessen:

- [ ] Branch `mainline-reduced` von `b10238` abzweigen (im selben Repo)
- [ ] Patches per `git cherry-pick -n` bzw. manuell aufsetzen, jeder mit Messung in der
      Commit-Message
- [ ] Alte Branches als Referenz behalten, nicht löschen

---

## Risiken

| Risiko | Auswirkung | Mitigation |
|---|---|---|
| Phase G misst unsauber (Taktschwankungen) | Falsche Entscheidung über 1600 Zeilen | Phase 0 der Hardware-Arbeit vorher abschließen |
| MTP-Verzicht kostet mehr als gedacht | Prefill-Regression im Serverbetrieb | Vor dem Fallenlassen mit realem Workload gegenmessen, nicht nur pp512 |
| Repack-GCN ist doch wertvoll | Verlust eines Fork-Vorteils | Phase 3 vor der Entscheidung, nicht danach |
| Upstream lehnt den nwarps-PR ab | Patch bleibt lokal | Kein Rückschritt — Aufwand ist ohnehin minimal |

---

## Querverweise

- [gfx906-naechste-optimierungsschritte.md](gfx906-naechste-optimierungsschritte.md) — Phase 0
  (Messbasis) ist Voraussetzung für Phase G
- [port-gfx906-kernels-from-skyne98.md](port-gfx906-kernels-from-skyne98.md) — Belege für den
  Nullbefund der Phasen 1–4
- [../MERGE_REPORT.md](../MERGE_REPORT.md) — MTP-Rauchtest und die Kopplungsproblematik
- [../AGENTS.md](../AGENTS.md) — Liste der Fork-Features, die zur Disposition stehen
