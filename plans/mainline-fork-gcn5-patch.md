# Ausführungsplan: Mainline forken + GCN5-MMQ-Patch

**Erstellt:** 2026-08-03 · **Status:** Schritte 0–4 erledigt (3.3/3.4 am 2026-08-05) · Schritt 6 offen
**Strategische Grundlage:** [fork-auf-mainline-reduzieren.md](fork-auf-mainline-reduzieren.md)

Ziel: ein Repo, das Mainline folgt und **genau einen** Patch trägt — die fehlende
MMQ-Konfiguration für GCN5/gfx906. TP, Repack, MTP-Schicht und die skyne98-Ports entfallen
(begründet und gemessen im Strategieplan).

---

## ✅ Ergebnis (2026-08-04)

Branch `gcn5`, vier Commits auf `b10238`. Eingriff in Upstream-Code: **6 Zeilen** in `mmq.cuh`
plus die neue Datei `mmq-config-gcn5.cuh`. Zum Vergleich: der alte Fork wich in **74 Dateien**
und ~12.400 Zeilen ab.

| Modell | Mainline | **gcn5-Patch** | Δ | alter Fork |
|---|---:|---:|---:|---:|
| Ornith-9B **Q8_0** | 684,72 | **768,76** | **+12,3 %** | 733,90 → Patch ist **4,7 % besser** |
| ornith-9B **Q5_K_M** | 581,09 | **668,54** | **+15,1 %** | 679,69 → Patch 1,6 % dahinter |
| Mistral-24B **Q6_K** | 173,26 | **259,85** | **+50,0 %** | nicht gemessen |

`test-backend-ops -o MUL_MAT` mit Typfilter: q4_K, q5_K, q6_K, q8_0 jeweils **2/2**.

**Der Patch besteht aus zwei Erkenntnissen:**

1. **`nthreads` 256 → 512.** gfx906 fiel auf die RDNA2-Config zurück; RDNA2 hat Wave-Größe 32,
   gfx906 hat 64 — also nur 4 statt 8 Warps pro Block.
2. **`stream_k` nur für K-Quants.** Das ist ein Tauschgeschäft, kein genereller Gewinn:

   | | Mainline | nthreads=512 | + stream_k |
   |---|---:|---:|---:|
   | Q5_K_M | 581,09 | 606,51 | **668,54** |
   | Q6_K | 173,26 | 228,01 | **259,85** |
   | Q8_0 | 684,72 | **769,43** | 718,48 |

   K-Quants gewinnen 10–14 %, Q8_0 **verliert 6 %**. Da die Config pro Typ gilt, lassen sich
   beide Optima gleichzeitig nehmen. Wäre nur Q5_K nachgemessen worden, hätte ein globaler
   Schalter Q8_0 still um 6,6 % verschlechtert.

**`occupancy`** wurde ebenfalls gesweept (1 vs. 2, CDNAs Wert) — **kein Effekt**, bleibt beim
RDNA2-Wert.

**Methodisch entscheidend war der Blick zu CDNA** statt blindem Variieren: die einzige andere
AMD-Architektur mit Wave-Größe 64 unterschied sich in genau zwei Feldern, eines davon war der
Treffer.

---

## Baseline (Referenz, 2026-08-03)

Single GPU (32-GB-Karte), `llama-bench -ngl 99 -fa 1 -r 3 -p 2048 -n 128`:

| Modell | Mainline `f2b52a87e` | Fork `aacf2aeb5` | Delta |
|---|---:|---:|---:|
| Ornith-9B **Q8_0** | 685,53 ± 0,55 | 733,90 ± 0,43 | **+7,1 %** |
| ornith-9B **Q5_K_M** | 580,44 ± 0,73 | 679,69 ± 0,80 | **+17,1 %** |

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

1. - [x] `test-backend-ops -o MUL_MAT` — q4_K, q5_K, q6_K, q8_0 je **2/2**.
       **Wichtig:** ein *ungefilterter* Lauf bricht bei `MUL_MAT(type_a=f32)` mit
       `CUBLAS_STATUS_INTERNAL_ERROR` ab — **unverändertes Mainline bricht am identischen
       Testfall ab**, also ein vorbestehendes hipBLAS-Problem dieser ROCm-Version, nicht der
       Patch. Ohne diese Kontrollmessung wäre der Patch fälschlich verdächtigt worden.
2. - [x] `llama-bench` auf echten Modellen — Q5_K_M, Q6_K, Q8_0, siehe Ergebnistabelle oben.
3. - [x] Inferenz-Rauchtest mit `llama-cli` (2026-08-05) — kohärenter Text, sauberer Exit,
       46,5 t/s Generation auf Q5_K_M.
       **Fallstrick:** `llama-cli` ist seit `b10240` eine Chat-UI. Ohne `-st`/`--single-turn`
       läuft sie über SSH (kein TTY) in eine Endlosschleife und schrieb 735 MB `> `-Prompts,
       bevor sie gestoppt wurde. Für Skripte immer `-no-cnv -st </dev/null` **plus** einen
       `head -c`-Deckel. `-no-cnv` allein genügt nicht.
4. - [x] Ausgabevergleich gegen Mainline (2026-08-05) — **Host/Device-Konsistenz belegt, aber
       nicht über Bit-Identität.** Siehe Abschnitt unten.

---

### Schritt 3.4 im Detail: Warum „Text identisch" das falsche Kriterium war

Aufbau: gleicher Worktree, `mmq.cuh` per `git checkout d5b7227c9 -- …` auf den Stand vor dem
Patch gesetzt, neu gebaut, gemessen, danach wiederhergestellt. Ein dritter Worktree schied aus
(nur 21 GB frei). Lauf jeweils `-s 1234 --temp 0 --top-k 1 -n 128`, identischer Prompt.

| Quant | `stream_k` | Mainline vs. gcn5 |
|---|---|---|
| **Q8_0** | `false` | **identisch, Zeichen für Zeichen** |
| **Q5_K_M** | `true` | weicht ab Zeile 8 ab |

**Kontrollmessung zuerst:** derselbe Build zweimal → identische Ausgabe. Die Abweichung stammt
also wirklich aus der Config und nicht aus Nichtdeterminismus.

**Damit ist die Ursache exakt isoliert.** `nthreads = 512` ist bit-exakt — Q8_0 durchläuft
denselben Patch und ändert nichts. Es hängt allein an `stream_k`: Stream-K zerlegt die
K-Dimension auf mehrere Blöcke und summiert Teilergebnisse zusammen. Andere Summationsreihenfolge
→ andere Gleitkomma-Rundung → bei Greedy-Decoding kippt ein knappes Token-Rennen anders.

**Das ist kein Host/Device-Divergenzfehler**, also nicht die Klasse, die dieser Schritt fangen
sollte. Ein solcher Fehler erzeugt Müll, NaNs oder einen Fault — nicht ein Synonym in einer sonst
sauberen, thematisch korrekten Antwort. Und er würde Q8_0 nicht verschonen.

**Das Kriterium war zu wörtlich formuliert.** Der Struct-Kommentar in `mmq.cuh` (~Z. 163) sagt
`// Should not affect results, only speed/…` — „**should** not", nicht „does not", und `stream_k`
steht selbst in genau diesem Struct. Bit-Identität kann diese Config gar nicht zusichern; gemeint
ist Korrektheit, nicht Reproduzierbarkeit auf Zeichenebene.

- [ ] **Rest-Offen:** Für die K-Quants fehlt ein Kriterium *mit Toleranz* statt Identität —
      Perplexity-Vergleich Mainline vs. gcn5. Promille-Abweichung → erledigt; sichtbare
      Abweichung → doch ein Rechenfehler. Braucht ein Testkorpus (z. B. `wikitext-2-raw`).

---

## Schritt 4: Tuning — ✅ erledigt

- [x] **`nthreads`:** 512 gesetzt. Ein Sweep über 1024 entfällt — das `CASE`-Makro enthält
      `static_assert(nthreads <= 512)`. Mainlines Design schließt also genau die Konfiguration
      aus, die im Legacy-Pfad den ungeklärten GPU-Fault auslöste (`nwarps=16`).
- [x] **`occupancy`:** 1 vs. 2 gesweept → kein Effekt (607,41 vs. 606,51). Bleibt bei 2.
- [x] **`stream_k`:** pro Typ gesetzt — für K-Quants an, sonst aus. Größter Einzelgewinn.
- [x] Werte im Code kommentiert (Modell, Test, Zahl, Datum)
- [ ] **Offen:** Q2_K, Q3_K, Q4_K haben `stream_k` per Analogie zu Q5_K/Q6_K bekommen,
      wurden aber nicht gemessen. `I`, `J` und `K_vram` sind unverändert von RDNA2 übernommen
      und für GCN5 nie untersucht.

---

## Schritt 5: Umstellung von Server und Betrieb

- [ ] Server-Checkout auf den neuen Branch, Build mit `scripts/mx-compile.sh`
- [ ] Rauchtests: `llama-server` mit einem Alltagsmodell, `-sm tensor` für Multi-GPU
- [ ] **`-tps` fällt weg** → Multi-GPU läuft künftig über upstreams `-sm tensor`.
      Startskripte und Doku entsprechend anpassen.
      ⚠ **Blockiert durch Schritt 6.3:** Ohne den Lane-Dispatch-Patch kostet das bei 8 GPUs
      möglicherweise bis zu 32 % Token-Generierung. Erst gegenmessen, dann umstellen.
- [ ] **MTP:** upstreams Basis bleibt nutzbar (`--spec-type draft-mtp`), nur die
      Fork-Optimierung (`LLAMA_ENABLE_MTP_OPT`) entfällt → ~19 % Prefill weniger bei
      MTP-Workloads. Vor der endgültigen Umstellung mit dem realen Server-Workload gegenmessen.
- [ ] AGENTS.md neu fassen: der Abschnitt „Fork-spezifische Features" schrumpft auf den
      GCN5-Patch

---

## Schritt 6: Nicht bewertete Patches aus dem Original-Fork (2026-08-05)

**Die Fork-Kette war bis hierher falsch angenommen.** Sie lautet:

`ggerganov/llama.cpp` → **`mxxm-t/mx-llama.cpp`** (Original) → `DENEB1312/mx-llama.cpp`
→ `stanus74/mx-llama.cpp` (`origin`)

`origin/master` steht auf dem **2026-07-06**, DENEB1312 auf dem **15.07.** Das Original ist seither
weitergezogen und hat vier Features, die der Strategieplan **nie bewertet hat** — sie waren zum
Zeitpunkt der Analyse schlicht nicht im Baum. Remotes dafür sind lokal angelegt: `mxorig`, `deneb`.

| Patch | Commit (`mxorig`) | Code-Zeilen | auf `gcn5` portierbar? |
|---|---|---:|---|
| BF16→F32 auf AMD ohne natives bf16 | `81a8712d0` (30.07.) | **14** | ✅ abhängigkeitsfrei |
| q8_1-Cache (quantized activation reuse) | `775a8051f` (04.08.) | 185 | ✅ reine CUDA-Dateien |
| Concurrent lane dispatch | `5d9efc8ca` (04.08.) | 223 | ✅ nur `ggml-backend-meta.cpp` (Upstream-Datei) |
| Whole-token graph capture | `751b6114c` (04.08.) | 561 | ❌ fasst `tp-allreduce.cu/.cuh` an |

**Entscheidend:** Der Token-Graph-Patch braucht das Custom-AllReduce-Subsystem, das dieser Plan
begründet verworfen hat — er ist ohne Rückholung von TP nicht übernehmbar. Die Trennlinie verläuft
damit **anders als in der `FEATURES.md` des Originals behauptet** („requires the concurrent lane
dispatch above"): Lane dispatch allein ist sehr wohl portierbar.

### Reihenfolge

> ⚠ **Messung 2026-08-05: BF16 ist für diesen Modellbestand wirkungslos.** Alle 14 GGUFs unter
> `~/data/models` geprüft (`gguf-py`, Tensortyp-Histogramm): **null BF16-Tensoren**, auch in den
> UD-/`*_XL`-Quants (`gemma-4-26B-A4B-it-UD-Q6_K_XL` 0/658, `gemma-4-31B-it-UD-Q6_K_XL` 0/833,
> `Qwen3-Coder-Next-UD-IQ4_XS` 0/843). Die nicht-quantisierten Tensoren sind **F32**. Der Patch
> prüft `compute_type == BF16` und feuert hier nie. Die „+18–19 % Prefill auf UD/`*_XL`" aus der
> `FEATURES.md` des Originals gelten für anders gepackte UD-Quants als die vorhandenen.
> **Konsequenz:** BF16 rutscht von Platz 1 auf zuletzt und bleibt nur als Upstream-PR interessant
> — die Lücke im GEMM-Pfad ist real, nur lokal nicht messbar. Reihenfolge daher: **q8_1-Cache
> zuerst**, dann Lane dispatch.

- [ ] **1. BF16 (`81a8712d0`) — nur als Upstream-PR, kein lokaler Nutzen.** Der Diff ist ein `else if`, symmetrisch zum bereits
      vorhandenen F16-Zweig in [ggml-cuda.cu](../ggml/src/ggml-cuda/ggml-cuda.cu) (`ggml_cuda_mul_mat_cublas`,
      ~Z. 1620): F16 hat einen Hardware-Fallback, **BF16 hat keinen**, also geht ein bf16-Tensor auf
      gfx906 ungebremst an rocBLAS. Dazu ein 10-Zeilen-Helper `fast_bf16_hardware_available()`.
      Ursache laut Commit-Message: *„a bf16 GEMM picks a 64x32x8 macro-tile and runs 3.5x slower
      than the F32 path on the same weights"*. Betrifft **alle** AMD-Karten vor CDNA/RDNA3, nicht nur
      gfx906 → besserer Upstream-PR-Kandidat als der MMQ-Patch selbst.
- [x] **2. q8_1-Cache (`775a8051f`) — übernommen 2026-08-05** als `77dad37f6`. Cherry-Pick, einziger
      Konflikt war `FEATURES.md` (trägt dieser Branch nicht); alle vier Quelldateien mergten
      automatisch. 175 Zeilen.

      **Korrektheit:** Greedy-Ausgabe Q8_0 **und** Q5_K_M zeichengleich mit/ohne Cache — die
      Bit-Exaktheit des Autors ist auf gfx906 reproduziert. Anders als `stream_k` verschiebt dieser
      Patch keine Rundung.

      **Nutzen (MI50, 1 GPU, `-r 3`, A/B über `GGML_CUDA_Q8_1_CACHE`):**

      | | Cache aus | Cache an | Δ |
      |---|---:|---:|---:|
      | Q5_K_M pp2048 | 665,82 ± 0,49 | 671,14 ± 1,24 | **+0,80 %** |
      | Q5_K_M tg128 | 46,32 ± 0,32 | 46,98 ± 0,17 | **+1,42 %** |
      | Q8_0 pp2048 | 769,67 ± 0,64 | 773,70 ± 0,74 | **+0,52 %** |
      | Q8_0 tg128 | 49,39 ± 0,16 | 49,92 ± 0,17 | **+1,07 %** |

      Alle vier Differenzen liegen außerhalb der Streuung, sind also echt — aber es sind
      **+0,5–1,4 % statt der angegebenen +2,2–2,6 %.** Der Autor hat auf **4 GPUs und einem
      MoE-Modell** gemessen; bei MoE teilen sich Router *und* Gate/Up dieselbe `ffn_norm`, es gibt
      dort also mehr wiederverwendbare Quantisierungen als in einem dichten 9B auf einer Karte.
      **Zweiter Beleg dafür, fremde Prozentzahlen nicht ungeprüft zu übernehmen** — diesmal stimmte
      die Richtung, nur nicht die Größe.

      - [ ] Offen: Gegenmessung auf einem MoE-Modell (`Qwopus3.6-35B-A3B`, `gemma-4-26B-A4B`),
            wo der Patch laut Herleitung deutlich mehr bringen müsste.
- [ ] **3. Lane dispatch (`5d9efc8ca`)** — **nur wenn der Server real `-sm tensor` über ≥2 GPUs fährt.**
      Laut Original +32 % TG auf 8 GPUs, +2,5 % auf 4, Prefill flat; inert bei einer GPU.
      **Das ist zugleich die Vorbedingung für Schritt 5:** dort fällt `-tps` weg, ohne dass der
      TG-Verlust je gemessen wurde. Bei 8 GPUs stehen bis zu 32 % im Raum — mehr als der gesamte
      MMQ-Gewinn.
- [ ] **4. Token graph (`751b6114c`)** — streichen, solange `gcn5` ohne TP-AllReduce bleibt.

Damit wüchse `gcn5` von 6 auf grob 420 Zeilen — zwei Größenordnungen unter den ~12.400 des alten
Forks, jeder Teil einzeln begründet und per Env-Var abschaltbar.

> ⚠ **Die Prozentzahlen sind Behauptungen eines fremden Repos, bis sie auf der eigenen Hardware
> nachgemessen sind.** Präzedenzfall: Repack war in diesem Fork dokumentiert und nachweislich toter
> Code (Phase 3). Das gilt besonders für die +32 %, an denen die Multi-GPU-Entscheidung hängt.

Zwei Punkte der `FEATURES.md` sind ohnehin schon Mainline: `--load-mode dio` und der Schalter
`GGML_CUDA_CUBLAS_COMPUTE_TYPE`. Wirklich fork-exklusiv ist nur `-tps`.

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
