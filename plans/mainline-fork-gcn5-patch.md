# Ausführungsplan: Mainline forken + GCN5-MMQ-Patch

**Erstellt:** 2026-08-03 · **Status:** ausführbar, noch nicht begonnen
**Strategische Grundlage:** [fork-auf-mainline-reduzieren.md](fork-auf-mainline-reduzieren.md)

Ziel: ein Repo, das Mainline folgt und **genau einen** Patch trägt — die fehlende
MMQ-Konfiguration für GCN5/gfx906. TP, Repack, MTP-Schicht und die skyne98-Ports entfallen
(begründet und gemessen im Strategieplan).

---

## Baseline (bereits gemessen, 2026-08-03)

Single GPU (32-GB-Karte), `llama-bench -ngl 99 -fa 1 -r 3 -p 2048 -n 128`:

| Modell | Mainline `f2b52a87e` | Fork `aacf2aeb5` | Delta |
|---|---:|---:|---:|
| Ornith-9B **Q8_0** | 685,53 ± 0,55 | 733,90 ± 0,43 | **+7,1 %** |
| ornith-9B **Q5_K_M** | 580,44 ± 0,73 | 679,69 ± 0,80 | **+17,1 %** |

Diese Fork-Werte sind das **Ziel**, das der Patch auf Mainline erreichen soll. Da Repack
nachweislich toter Code ist (Phase 3), stammt der gesamte Vorsprung aus dem MMQ-Tuning —
der Patch sollte ihn also vollständig reproduzieren können.

---

## Schritt 0: Repo und Branch

Kein neues Repository. Der Branch entsteht im vorhandenen Repo, damit die Historie mit allen
Messungen zugänglich bleibt.

```bash
git fetch upstream --tags
git checkout -b gcn5 b10238        # oder der dann aktuelle Release-Tag
```

- [ ] Branch `gcn5` von `b10238` anlegen
- [ ] `.github/workflows`: nur die self-hosted/HIP-Workflows aktiv lassen
      (AGENTS.md-Konvention, `.yml.disabled` für den Rest)
- [ ] Aus dem alten Fork **mitnehmen** (reine Zusatzdateien, kein Eingriff in Upstream-Code):
  - `scripts/mx-compile.sh`, `scripts/bench-gfx906-nwarps.sh`
  - `docs/gfx906/**`
  - `plans/**`, `AGENTS.md` (auf den neuen Stand anpassen)
- [ ] **Nicht** mitnehmen: `ggml/src/ggml-cuda/gfx906/**`, `repack-gcn.*`, `tp-allreduce.cu`,
      MTP-Schicht, `mmq.cuh`-Divergenz

---

## Schritt 1: Baseline auf dem neuen Branch verifizieren

Vor dem Patch, damit der Vergleich sauber ist.

```bash
scripts/mx-compile.sh
HIP_VISIBLE_DEVICES=1 build/bin/llama-bench \
  -m ~/data/models/ornith-1.0-9b-Q5_K_M.gguf -ngl 99 -fa 1 -r 3 -p 2048 -n 128
```

- [ ] Erwartung: ~580 t/s (Q5_K), ~685 t/s (Q8_0) — muss zu den Zahlen oben passen
- [ ] Weicht es ab, erst die Ursache klären, nicht patchen

---

## Schritt 2: Der Patch

Mainline wählt die MMQ-Konfiguration **pro Architektur**; gfx906 (GCN5) hat keinen eigenen
Fall und fällt bis `ggml_cuda_mmq_get_config_rdna2` durch — eine RDNA2-Konfiguration auf einer
Karte mit doppelter Wave-Größe (64 statt 32). Genau das ist die Unterdimensionierung, die der
alte Fork über `nwarps`-Defines behoben hat.

**Drei Berührungspunkte** — die vorhandenen Makros reichen aus, es braucht keine neuen:

1. **Neue Datei** `ggml/src/ggml-cuda/mmq-config-gcn5.cuh`, analog zu den vorhandenen
   `mmq-config-rdna2.cuh` / `-cdna.cuh`. Sinnvoller Startpunkt: von `rdna2` abgeleitet, aber
   `nthreads = 512` (= 8 Warps × 64 Lanes, entspricht dem gemessenen Optimum des alten Forks).
2. **Host-Dispatch** in `mmq.cuh` (~Z. 228): vor dem RDNA2-Fallback einhängen, mit dem bereits
   vorhandenen `GGML_CUDA_CC_IS_GCN(cc)` (`common.cuh:91`).
3. **Device-Dispatch** in `mmq.cuh` (~Z. 253): derselbe Zweig über `__gfx906__` — dieses Makro
   wird in Mainlines `common.cuh` bereits verwendet (Z. 705, 752).

> ⚠ **Host und Device MÜSSEN übereinstimmen.** Das ist dieselbe Falle, die der alte Fork in
> seinem `mmq.cuh`-Kommentar dokumentiert: die Konfiguration geht in `__launch_bounds__` und in
> die Tile-Indizierung ein. Eine Abweichung zwischen beiden Pfaden erzeugt keinen Compilefehler,
> sondern falsche Ergebnisse oder GPU-Faults.

- [ ] `mmq-config-gcn5.cuh` anlegen
- [ ] Host-Dispatch ergänzen
- [ ] Device-Dispatch ergänzen
- [ ] `CMakeLists.txt` prüfen (Header werden i. d. R. automatisch erfasst)

---

## Schritt 3: Verifikationskette (verbindlich, in dieser Reihenfolge)

Die Reihenfolge ist Ergebnis der Fehlschlüsse vom 2026-08-03 — jeder Schritt fängt eine
Fehlerklasse ab, die der vorherige nicht sieht.

1. - [ ] `test-backend-ops -o MUL_MAT` — **notwendig, nicht hinreichend.** Das Gate hat den
       `nwarps=16`-Fault mit 2/2 durchgewinkt.
2. - [ ] `llama-bench` auf **echten Modellen** (Q5_K_M und Q8_0) — deckt die Shapes ab, die
       das Gate nicht trifft. Hier muss der Gewinn sichtbar werden.
3. - [ ] Inferenz-Rauchtest mit `llama-cli` — deckt Semantikfehler ab, die weder Gate noch
       Benchmark sehen.
4. - [ ] Ausgabe gegen den Mainline-Build gegenprüfen (gleicher Seed, gleicher Prompt):
       **Text muss identisch sein.** Die Konfiguration darf laut Struktur-Kommentar
       ausschließlich Geschwindigkeit beeinflussen, nie Ergebnisse.

---

## Schritt 4: Tuning

`nthreads = 512` ist eine **Ableitung** aus dem alten Fork, keine Messung auf Mainlines
MMQ-Implementierung. Die übrigen Felder (`I`, `J`, `occupancy`, `stream_k`, `K_vram`) sind
bisher überhaupt nicht für GCN5 untersucht.

- [ ] `nthreads` sweepen: 256 / 512 / 1024 — je Konfiguration Gate + `llama-bench`
- [ ] **Achtung:** 1024 Threads entspricht dem alten `nwarps=16`, das im Legacy-Pfad einen
      GPU-Fault auslöste (Ursache bis heute ungeklärt, siehe
      `gfx906-naechste-optimierungsschritte.md` B.2). Falls es auch hier faultet: nicht
      erneut tagelang suchen, sondern ausschließen und weitergehen.
- [ ] Danach optional `I`/`J`/`stream_k` variieren — nur mit je einer Messung als Begründung
- [ ] Jeden gewählten Wert im Code kommentieren: **Modell, Test, Zahl, Datum**

---

## Schritt 5: Umstellung von Server und Betrieb

- [ ] Server-Checkout auf den neuen Branch, Build mit `scripts/mx-compile.sh`
- [ ] Rauchtests: `llama-server` mit einem Alltagsmodell, `-sm tensor` für Multi-GPU
- [ ] **`-tps` fällt weg** → Multi-GPU läuft künftig über upstreams `-sm tensor`.
      Startskripte und Doku entsprechend anpassen.
- [ ] **MTP:** upstreams Basis bleibt nutzbar (`--spec-type draft-mtp`), nur die
      Fork-Optimierung (`LLAMA_ENABLE_MTP_OPT`) entfällt → ~19 % Prefill weniger bei
      MTP-Workloads. Vor der endgültigen Umstellung mit dem realen Server-Workload gegenmessen.
- [ ] AGENTS.md neu fassen: der Abschnitt „Fork-spezifische Features" schrumpft auf den
      GCN5-Patch

---

## Erfolgs- und Abbruchkriterien

**Erfolg:** Q5_K ≥ 670 t/s und Q8_0 ≥ 725 t/s bei pp2048 (also im Bereich der alten
Fork-Werte), Gate sauber, Ausgabe identisch zu Mainline.

**Teilerfolg:** Gewinn vorhanden, aber kleiner als im alten Fork → akzeptieren und
dokumentieren; Mainlines MMQ ist ein anderer Kernel, ein exakter Gleichstand ist nicht
garantiert.

**Abbruch:** Kein messbarer Gewinn gegenüber der RDNA2-Fallback-Konfiguration. Dann ist auch
der letzte Patch hinfällig — und die richtige Antwort lautet, unverändertes Mainline zu
benutzen. Das wäre kein Scheitern, sondern das billigstmögliche Ergebnis.

---

## Danach: Upstream-PR erwägen

Eine fehlende Architektur-Konfiguration ist eine Lücke, kein Sonderwunsch — die Chance auf
Aufnahme ist real, und bei Aufnahme sinkt der Pflegeaufwand auf **null**.

- [ ] Messwerte für zwei Modelle/Quantisierungen beilegen
- [ ] Upstream hat eine Anti-AI-PR-Policy: die Herleitung als eigene Messung darstellen,
      nicht als Werkzeugausgabe
- [ ] Bis zur Aufnahme (oder Ablehnung) bleibt der Patch lokal auf `gcn5`

---

## Laufender Betrieb: Rebase statt Merge

```bash
git fetch upstream --tags
git rebase --onto b10xyz b10238 gcn5
```

`git log b10238..gcn5` zeigt dann jederzeit exakt den eigenen Anteil. Kadenz: ein Release alle
zwei bis vier Wochen genügt.

---

## Risiken

| Risiko | Auswirkung | Mitigation |
|---|---|---|
| Host/Device-Config divergiert | Falsche Ergebnisse oder GPU-Fault, **kein** Compilefehler | Beide Zweige in einem Commit ändern; Schritt 3.4 (Ausgabevergleich) |
| Mainlines MMQ profitiert anders als der Legacy-Pfad | Gewinn kleiner als +17 % | Schritt 4 (Sweep) statt Übernahme des alten Werts |
| Verlust der MTP-Optimierung fällt im Betrieb stärker auf als gedacht | Prefill-Regression im Server | Vor der Umstellung mit realem Workload gegenmessen |
| Taktschwankungen verfälschen den Sweep | Falsche Parameterwahl | Phase 0 aus `gfx906-naechste-optimierungsschritte.md` vorher abschließen |
