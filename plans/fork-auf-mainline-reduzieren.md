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
| MMQ-`nwarps` 4→8 | **~15 Zeilen** | **+19 % … +28 % pp** | `mmq.cuh`-Kommentar, Sweep 2026-08-03 |
| Multi-Stage TP (`-tps`) | ~885 + 728 Z. | **+32 % pp** | Nutzer-Messung, Baseline unklar → Gate |
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

- [ ] Identisches Modell, identische Flags, drei Konfigurationen auf derselben Maschine:
  1. Mainline (`/opt/llama.cpp`, `b10238`) **ohne** TP — Referenz
  2. Mainline **mit** `-sm tensor`
  3. Fork mit `-sm tensor -tps 2`
- [ ] `llama-bench`, ≥ 5 Wiederholungen, pp2048 und pp4096, Median
- [ ] Taktprotokoll mitschreiben (siehe `gfx906-naechste-optimierungsschritte.md` Phase 0)

**Entscheidungsregel:**

| Ergebnis | Konsequenz |
|---|---|
| Mainline-TP ≈ Fork-TP | Multi-Stage-TP **nicht** portieren — größter Einzelgewinn an Wartbarkeit |
| Fork-TP deutlich besser | Portieren, aber als isolierter, klar abgegrenzter Patch |

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

## Phase 2: TP — nur falls Phase G es rechtfertigt

- [ ] Abhängig von Phase G, sonst überspringen
- [ ] Bei Portierung: `-tps`-Flag, `tp-allreduce.cu`, Meta-Device-Anteile in
      `ggml-backend-meta.cpp` als **ein** zusammenhängender Patch, nicht verstreut

---

## Phase 3: GCN-Repack — erst messen, dann entscheiden

2283 Zeilen ohne isolierte Messung sind der größte blinde Fleck des Forks.

- [ ] A/B über den vorhandenen Schalter bzw. `ggml_backend_buft_is_cuda_repack`-Pfad
- [ ] Ohne belegten Gewinn: **nicht** übernehmen

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
