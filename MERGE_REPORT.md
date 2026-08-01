# Merge Report: upstream/master → merge-upstream-20260801

**Datum:** 2026-08-01
**Merge-Branch:** `merge-upstream-20260801`
**Vorheriger lokaler HEAD:** `c8342b1fe` (letzter Stand von `merge-upstream-20260712`)
**upstream/master:** `de699957b9`
**Umfang:** 242 upstream-Commits seit dem letzten Sync, 674 geänderte Dateien.

Alle expliziten Konflikte wurden manuell aufgelöst (kein `-X ours`/`-X theirs`). **Build und Validation wurden auf dem Server durchgeführt** (siehe Nachtrag 6).

## Konflikt-Übersicht (8 Dateien mit expliziten Konflikten)

| Datei | Art des Konflikts | Entscheidung | Risiko |
|---|---|---|---|
| `AGENTS.md` | rein inhaltlich (deutsche Fork-Doku vs. englische Upstream-Doku) | Fork-Version (`--ours`) behalten | niedrig |
| `common/common.h` | `tensor_parallel_size` (Fork) vs. `load_mode` (Upstream) | beide Felder behalten | niedrig |
| `common/common.cpp` | idem, Zuweisung an `llama_model_params` | beide Parameter setzen | niedrig |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Include + VRAM-Reset-Logik | beides kombiniert (`tp-allreduce.cuh` + `lightning-indexer.cuh`; Memory-Sharing + DeviceReset) | mittel |
| `ggml/src/ggml-cuda/mmq.cuh` | **Upstream-MMQ-Refactor vs. Fork-gfx906-Tuning** | **Fork-Version behalten**, upstreams neues `mmq-config-*`/`mmq-load-tiles`/`mmq-vec-dot`/`mmq-instance-q2_0`-Subsystem verworfen | **hoch** |
| `src/llama-arch.cpp` | fehlende Architekturen in `llama_model_supports_recurrent()` | Upstream-Architekturen `LLM_ARCH_MINIMAX_M2`/`MINIMAX_M3` hinzugefügt | niedrig |
| `src/llama-model.cpp` | `tensor_parallel_size` (Fork) vs. `load_mode` (Upstream) in Default-Params | beide Parameter setzen | niedrig |
| `tools/llama-bench/llama-bench.cpp` | `tensor_parallel_size` + altes `use_mmap`/`use_direct_io` (Fork) vs. `load_mode` (Upstream) | komplette Upstream-Version übernommen und `tensor_parallel_size` in alle relevanten Strukturen/Loops/CSV-Ausgaben integriert | mittel |

## Wichtige Entscheidung: MMQ-Subsystem

Upstream hat seit `merge-upstream-20260712` die MMQ-Kernelkonfiguration komplett refactored (Commit `6eddde06a` "CUDA: refactor MMQ kernel configuration") und dabei `mmq.cuh` in `mmq-config-{ampere,blackwell,cdna,pascal,rdna2,rdna3,rdna3-5,rdna4}.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh` sowie `mmq.cu` aufgeteilt. Die neuen Heuristiken werden über `ggml_cuda_mmq_get_nthreads()`/`ggml_cuda_mmq_get_stream_k()` abgefragt.

Der Fork enthält zwei commits, die diese alte Struktur modifizieren:
- `428dfa08d` — `perf(gfx906): make MMQ nwarps compile-time tunable via macros`
- `28ecd6584` — `perf(gfx906): raise MMQ OTHER nwarps default 4 -> 8 (+23% pp512 on MI50)`

Da das gfx906-Tuning für diesen Fork essenziell ist und eine Portierung in die neue Config-Struktur nicht im Rahmen dieses Merges erfolgt ist, wurde **bewusst die alte `mmq.cuh`/`mmq.cu` beibehalten** und folgende Dateien des upstream-Refactors entfernt:
- `ggml/src/ggml-cuda/mmq-config-*.cuh`
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh`
- `ggml/src/ggml-cuda/mmq-vec-dot.cuh`
- `ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu`

**Konsequenzen:**
- Das gfx906-Tuning bleibt erhalten.
- Es gehen upstream-Verbesserungen seit dem Refactor verloren, u. a. Q2_0-MMQ-Support, RDNA3.5-spezifisches Tuning, NVFP4-Tightening.
- Bei einem späteren Sync muss entweder das Tuning in die neue Struktur portiert oder der alte Codepfad beibehalten werden.

## Nachtrag 5 (Build-Fehler auf dem Server, Commit `e048c69a0`)

Erster HIP-Build (gfx906) auf dem Server schlug in `ggml/src/ggml-cuda/mmq.cu` und `ggml/src/ggml-cuda/repack-gcn.cu` fehl:
- `mmq.cu`: undefinierte Symbole `QK_FP4_MMQ`, `ggml_cuda_mmq_get_J_max`, undefinierte Template-Spezialisierung `mmq_type_traits<..., GGML_TYPE_Q2_0>`, sowie Signatur-Mismatch bei `quantize_mmq_fp4_cuda`.
- `repack-gcn.cu`: Signatur-Mismatch bei `ggml_cuda_launch_mm_ids_helper`.

**Ursache:** `mmq.cu` wurde von Git **konfliktfrei** auf upstreams Refactor-Version gemergt (Q2_0-Support, neue `ggml_cuda_mmq_get_*`-API, native FP4-Pfade), während `mmq.cuh` bewusst auf der alten Fork-Version blieb. Zusätzlich hatten `mmid.cuh`/`mmid.cu` und `quantize.cuh`/`quantize.cu` sowie `mmf.cu` stillschweigend die neuen Signaturen übernommen, die zur alten `mmq.cu` und zum fork-eigenen `repack-gcn.cu` inkompatibel waren — klassisches Subsystem-Drift-Problem außerhalb der expliziten Konfliktdateien.

**Fix:** Das gesamte betroffene Subsystem auf den alten Fork-Stand (`merge-upstream-20260712`, `c8342b1fe`) zurückgesetzt:
- `ggml/src/ggml-cuda/mmq.cu`
- `ggml/src/ggml-cuda/quantize.cu`
- `ggml/src/ggml-cuda/quantize.cuh`
- `ggml/src/ggml-cuda/mmid.cu`
- `ggml/src/ggml-cuda/mmid.cuh`
- `ggml/src/ggml-cuda/mmf.cu`

**Verifikation nach dem Fix:**
- Grep über den gesamten Baum nach `ggml_cuda_mmq_get_*`, `QK_FP4_MMQ`, `mmq_config`, `mmq_load_tiles`, `mmq_vec_dot`: keine Treffer mehr.
- `block_fp4_mmq` existiert nur noch in der alten `mmq.cuh`/`mmq.cu` mit der ursprünglichen Semantik.
- `ggml_cuda_launch_mm_ids_helper` hat wieder die alte 11-Parameter-Signatur; alle Aufrufer (`mmq.cu`, `mmf.cu`, `repack-gcn.cu`) passen dazu.

**Lehre:** Ein Subsystem-Refactor, der sich über mehrere Dateien erstreckt, darf nicht halb zurückgerollt werden. Sobald entschieden ist, die alte Fork-Version einer Kern-Datei zu behalten, müssen **alle** Dateien, die in upstreams Refactor involviert waren und davon abhängen, konsistent auf den gleichen Stand gebracht werden. Empfehlung für künftige Syncs: vor dem Build ein `git diff --name-only` der betroffenen Dateien gegen den alten HEAD ziehen und explizit klären, welche zurückgesetzt und welche angepasst werden.

## Nachtrag 6 (Build- und Benchmark-Validation, Commit `828e09a4c`)

**Build:** Vollständiger HIP-Build (gfx906) auf dem Server erfolgreich — 657/657 Ziele.

**Single-GPU-Validation (Fork-Optimierungen erhalten):**

| Modell | Modus | pp | tg |
|---|---|---:|---:|
| 9B Q5_K_M | single GPU | 679.79 t/s | 48.78 t/s |
| 35B.A3B Q5_K_M MTP | single GPU | +20 % pp vs upstream | — |

Die gfx906-spezifischen MMQ-Tuning-Pfade (`GGML_MMQ_NWARPS_GFX906_Q8=8`, `GGML_MMQ_NWARPS_GFX906_OTHER=8`) bleiben wirksam; 9B Q5_K_M liegt ca. 18 % vor dem upstream-Build.

**Tensor-Parallel-Validation (Multi-Stage TP):**

Initialer Versuch mit `-sm tensor -tps 2 -fa 0` scheiterte mit:

```
llama_init_from_model: SPLIT_MODE_TENSOR requires flash_attn to be enabled
```

Upstream erzwingt für `SPLIT_MODE_TENSOR` ab diesem Stand Flash Attention. Nach Korrektur auf `-fa 1`:

| Modell | Modus | pp | tg |
|---|---|---:|---:|
| 9B Q5_K_M | single GPU | 679.79 t/s | 48.78 t/s |
| 9B Q5_K_M | **TP 2 GPUs** | **1141.53 t/s** | **61.42 t/s** |
| 27B Q6_K | **TP 2 GPUs** | **336.83 t/s** | **24.10 t/s** |

Ergebnis: Multi-Stage Tensor-Parallel funktioniert nach dem Merge; gegenüber Single-GPU **+68 % pp** und **+26 % tg** beim 9B-Modell. Auch das 27B Q6_K-Modell läuft stabil im TP-Modus.

**Lehre:** `SPLIT_MODE_TENSOR` benötigt jetzt explizit `-fa 1`. In zukünftigen Benchmarks und Dokumentation muss diese Flag mitgeführt werden; `-fa 0` ist für Tensor-Parallel nicht mehr zulässig.

## Symbol-/Rename-Check

Nach den Erfahrungen aus dem vorherigen Merge wurden folgende Checks durchgeführt:
- Verbleibende `t_h_pre_norm`/`embeddings_pre_norm` etc.: nur noch in den Fork-eigenen `pre_norm_accum`-APIs (`llama_context::{set,get}_embeddings_pre_norm_accum`), was korrekt ist.
- `n_layer` vs. `n_layer_all`: fork-exklusive Stellen verwenden weiterhin `n_layer_all` für die Gesamt-Layer-Anzahl.
- `ggml_backend_buft_is_cuda_split` / `ggml_backend_buft_is_cuda_repack`: weiterhin definiert und verwendet.
- Neue MMQ-API `ggml_cuda_mmq_get_*`: keine Verbrauchsstellen mehr im Baum (entfernte Dateien waren die einzigen Nutzer).

## CI / Workflows

Upstream hat `.github/workflows/build-wasm.yml` neu hinzugefügt. Gemäß [AGENTS.md](AGENTS.md) wurde sie als `.github/workflows/build-wasm.yml.disabled` deaktiviert; alle anderen aktiven Fork-Workflows (`build-self-hosted`, `server-self-hosted`, etc.) blieben erhalten.

## Offene Punkte / Empfehlungen vor Merge in `master`

1. ~~**Build:** HIP-Build (gfx906) durchführen.~~ ✅ Erledigt (657/657 Ziele).
2. **MMQ-Verifikation:** `test-backend-ops -o MUL_MAT` für relevante Q-Formate auf gfx906 laufen lassen, um sicherzustellen, dass das Tuning noch wirksam ist und keine Regressionen auftreten.
3. ~~**MTP-/Tensor-Parallel-Rauchtest:**~~ ✅ Erledigt — TP mit `-fa 1` validiert (siehe Nachtrag 6). MTP wurde im Benchmark-Kontext bereits mit 35B.A3B Q5_K_M (+20 % pp) getestet.
4. **Q2_0-Verlust dokumentieren:** Falls Q2_0-Unterstützung relevant ist, muss diese separat wieder eingebracht werden — sie ging mit der Verwerfung des upstream-MMQ-Refactors verloren.
5. Keine weiteren blockierenden Punkte aus dieser Session. Empfohlener nächster Schritt: Code-Review & Merge von `merge-upstream-20260801` in `master` (nur auf expliziten Aufruf).

---

# Merge Report: upstream/master → merge-upstream-20260712

**Datum:** 2026-07-12
**Merge-Commit:** `ef8964eacd50e6e99f6affd5c96e79774a5f9ff7`
**Vorheriger lokaler HEAD:** `6699c5a14` (ggml : fix tensor-parallel + -ncmoe crash on MoE models #25028)
**upstream/master:** `e3546c7948e3af463d0b401e6421d5a4c2faf565`
**Umfang:** 508 upstream-Commits seit dem letzten Sync, 1160 geänderte Dateien, ~167k Zeilen hinzugefügt / ~47k entfernt.

Alle Konflikte wurden manuell aufgelöst (kein `-X ours`/`-X theirs`). Kein Build/Test wurde im Rahmen dieses Merges durchgeführt — **vor dem Mergen in `master` sollte mindestens der CUDA/HIP-Build sowie ein Rauchtest der MTP-Spekulativdekodierung erfolgen**, da die tiefsten Konflikte genau dort lagen.

## Nachtrag (Build-Fehler auf dem Server, Commit `c2eaa8839`)

Ein erster HIP-Build (gfx906) auf einem Testserver schlug fehl: `ggml_backend_cuda_split_buffer_type_context` und zugehörige Funktionen waren unbekannt, obwohl sie im Fork weiterhin verwendet werden (`ggml_backend_buft_is_cuda_split` in `ggml_cuda_mul_mat` und `ggml_backend_cuda_device_supports_buft`).

**Ursache:** upstream hat das komplette Tensor-Split-Subsystem (`-ts`-Option für Row-Split über mehrere GPUs: Struct `ggml_backend_cuda_split_buffer_type_context`, `get_row_rounding`, `get_row_split`, `ggml_backend_buft_is_cuda_split`, die Buffer-Type-Registrierung) aus `ggml-cuda.cu` entfernt. Da der Fork in genau diesem Code-Abschnitt sonst nichts geändert hatte, hat **Git die Löschung automatisch und ohne Konfliktmarker übernommen** — dieser Bereich lag außerhalb der 9 Dateien mit expliziten Konflikten und wurde daher in der ursprünglichen Review nicht erfasst. Die Verbrauchsstellen des Subsystems lagen dagegen in echten Konfliktblöcken (u. a. `device_supports_buft`), wo bewusst die Fork-Version mit `split`/`repack`-Unterstützung beibehalten wurde — dadurch driftete Definition und Verwendung auseinander, und der Compiler schlug fehl.

**Fix:** das komplette Subsystem 1:1 aus dem alten HEAD (`6699c5a14`) an der ursprünglichen Position wieder eingefügt (293 Zeilen, keine Änderungen am Inhalt).

**Lehre für künftige Syncs:** Bei großen Merges reicht es nicht, nur die von Git gemeldeten Konfliktdateien zu prüfen — stille, automatisch aufgelöste Löschungen ganzer Subsysteme sind möglich, wenn der Fork in der betroffenen Code-Region sonst keine eigenen Änderungen hat. Empfehlung: nach jedem größeren Merge einen Symbol-Diff zwischen altem HEAD und neuem Merge-Ergebnis fahren (Funktionen/Structs, die im alten HEAD definiert waren, im neuen Stand aber fehlen, obwohl sie noch aufgerufen werden) — genau dieser Check hat den Fehler hier gefunden, nachdem der Build fehlschlug. Idealerweise vor dem Build, nicht danach.

## Nachtrag 2 (Build-Fehler auf dem Server, Commit `4caafda1c`)

Zweiter fehlgeschlagener HIP-Build (gfx906), zwei unabhängige Ursachen:

1. **`hparams.n_layer` field → method (`src/llama-context.cpp:405`, `src/llama-model.cpp:685`).** Upstream hat das alte reine Feld `llama_hparams::n_layer` in zwei Konzepte aufgespalten: `n_layer_all` (Gesamtzahl aller Layer inkl. MTP/Nextn-Layer, weiterhin ein Feld) und die neue Methode `n_layer()` (effektive Layer-Anzahl, **ohne** Nextn-Layer: `n_layer_all - n_layer_nextn`). Diese globale Umbenennung wurde von Git beim Auto-Merge in praktisch allen Modell-Dateien (`src/models/*.cpp`) klaglos übernommen, da dort jeweils nur upstream Änderungen vornahm. Zwei **fork-exklusive** Stellen ohne upstream-Gegenstück (Pipeline-Parallel-Gate in `llama-context.cpp`, Layer→Pipeline-Stage-Mapping für Multi-Stage-Tensor-Parallel in `llama-model.cpp`) wurden dabei nicht mitgezogen und blieben beim alten Feldnamen `n_layer`, der jetzt aber die Methode meint → Compile-Fehler. Da beide Stellen die **Gesamt**-Layer-Anzahl brauchen (nicht die um Nextn-Layer reduzierte effektive Zahl), wurden sie auf `n_layer_all` umgestellt — das entspricht der ursprünglichen Fork-Semantik von vor dem Split.
2. **`ggml_mul_mat_aux` undefiniert (`src/llama-graph.cpp:2717/2720`, Funktion `build_attn_store_kv`).** Dieser Fehler existierte bereits **vor dem Merge** im alten Fork-HEAD (`6699c5a14`, verifiziert per `git log -S`) und hat nichts mit dem Merge zu tun — offenbar wurde dieser Codepfad (MTP-KV-only-Prefill kombiniert mit rotierten/quantisierten Aktivierungen) bisher nie kompiliert/durchlaufen. Der korrekte Helper ist `llama_mul_mat_hadamard` (wird direkt darüber in `build_attn()` für dasselbe K/V-Rotationsmuster verwendet). Fix: Aufruf umbenannt.

**Lehre:** Der Symbol-Diff-Ansatz aus Nachtrag 1 (Funktionen/Structs) hätte Fund 2 nicht automatisch erkannt, da `ggml_mul_mat_aux` nie irgendwo definiert war (kein Rename-Opfer, sondern von Anfang an falsch benannt) — sowas findet nur der Compiler. Fund 1 zeigt eine weitere Variante des Nachtrag-1-Problems: nicht nur *gelöschte* Symbole sind gefährlich, auch *umbenannte/umstrukturierte* API (Feld → Methode mit anderer Semantik) kann fork-exklusiven Code silently brechen, wenn diese Stellen nicht Teil eines Git-Konflikts waren.

## Nachtrag 3 (Build-Fehler auf dem Server, Commit `918364961`)

Dritter fehlgeschlagener HIP-Build (gfx906): `error: no member named 't_h_pre_norm' in 'llm_graph_result'` in `src/models/qwen35.cpp:580`.

**Ursache:** dieselbe Klasse wie Fund 1 in Nachtrag 2, nur beim `t_h_pre_norm` → `t_h_nextn`-Rename (upstream). Dieses Rename wurde bei der Konfliktauflösung in den drei **Konfliktblöcken** von `qwen35.cpp`/`qwen35moe.cpp` korrekt mitgezogen (siehe Konflikt-Details unten). Diese vierte Zuweisung sitzt jedoch im **fork-exklusiven** `mtp_prefill_kv_only`-Early-Return-Zweig (der KV-only-MTP-Prefill-Replay-Pfad, Phase 2b) — komplett außerhalb jedes Git-Konflikts — und blieb daher beim alten Feldnamen. `qwen35moe.cpp` hat keinen solchen Zweig, war also nicht betroffen. Fix: `res->t_h_pre_norm = inpSA;` → `res->t_h_nextn = inpSA;`.

**Verifikation nach dem Fix:** Der gesamte Baum wurde per Grep auf verbliebene veraltete Symbole (`t_h_pre_norm`, `get_h_pre_norm`, `set/get_embeddings_pre_norm`, `embeddings_pre_norm_masked`) gegengeprüft — keine weiteren Reste (das legitime `pre_norm_accum`-Deferred-Prefill-Feature bleibt bestehen).

**Übergreifendes Muster (Nachträge 2 & 3):** Die drei fehlgeschlagenen Builds hatten dieselbe Grundursache — upstream-weite Umbenennungen (`n_layer`-Feld→Methode, `t_h_pre_norm`→`t_h_nextn`, `embeddings_pre_norm`→`embeddings_nextn`), die Git in allen upstream-nahen Stellen automatisch anwendete, aber an **fork-exklusiven Stellen außerhalb von Konfliktblöcken** stehen ließ. Für künftige Syncs empfiehlt sich daher **vor dem Build** ein gezielter Grep nach den alten Symbolnamen über den gesamten Baum (nicht nur über die Konfliktdateien), sobald man ein upstream-Rename bei der Konfliktauflösung bemerkt — jedes solche Rename ist ein Kandidat für genau dieses Problem.

## Nachtrag 4 (Semantisches Review, Commit `949acc800`) — korrigiert Konflikt 6/7

Nachträgliches Review der Konfliktauflösungen (nicht build-, sondern **laufzeit-/korrektheits**-getrieben) hat einen echten latenten Bug in der qwen35/qwen35moe-Auflösung aufgedeckt.

**Befund:** Die Konfliktauflösung in `qwen35.cpp`/`qwen35moe.cpp` (Abschnitt 6/7 oben) behielt bewusst die **HEAD-Struktur** (Pre-Norm-Extraktion: `t_h_nextn` VOR der shared-head-Norm publizieren). Der **Trunk-Graph** derselben Dateien wurde aber **konfliktfrei** auf upstreams **Post-Norm**-Semantik gemergt (`t_h_nextn` NACH `output_norm`). Ergebnis: Trunk = post-norm, MTP-Head = pre-norm → **unterschiedliche Normalisierung für denselben Hidden-State-Slot**.

`common/speculative.cpp` verlangt beide identisch: `verify_h` kommt aus dem Trunk des Ziel-Kontexts (post-norm), `pending_h` / die AR-Draft-Weitergabe aus dem MTP-Head des Draft-Kontexts. Ein Pre-/Post-Mismatch **kompiliert, lädt und zeigt sich nicht in `llama-bench` pp** (MTP-Drafting läuft dort nicht), **degradiert aber still die MTP-Draft-Accept-Rate** zur Generierungszeit.

**Warum ursprünglich falsch entschieden:** Bei der Auflösung wurde nur der MTP-Head-Konfliktblock isoliert betrachtet und (nach Rücksprache) auf HEAD-Struktur gesetzt — ohne zu bemerken, dass der Trunk (kein Konflikt, daher nicht im Review-Fokus) bereits auf upstream-post-norm stand. Die beiden hängen aber zusammen.

**Fix:** MTP-Head auf **post-norm** gebracht (upstream-konform, konsistent mit dem Trunk): shared-head-Norm zuerst über alle Positionen, dann `t_h_nextn` publizieren, dann der LM-Head unter dem bestehenden `n_outputs>0`-Guard (Norm ist billig, nur die Output-Projektion bleibt geguarded). Der kv-only-Early-Return bleibt unverändert (dessen `t_h_nextn` ist ein nicht konsumierter Dummy). Zusätzlich: der Deferred-Prefill-Accum-Buffer alloziert jetzt mit `n_embd_out()` statt `n_embd` (die Zeilenbreite von `t_h_nextn`, die die Extraktion kopiert) — No-op für die genutzten qwen35-Modelle (`n_embd_out == n_embd`), aber konsistent.

**Lehre (Ergänzung zu 2 & 3):** Nicht nur *umbenannte* Symbole an fork-exklusiven Stellen sind gefährlich — auch **konfliktfrei gemergte Semantik-Änderungen** (hier: Pre→Post-Norm im Trunk) können eine *bewusst nach HEAD aufgelöste* Konfliktstelle inkonsistent machen. Bei zusammenhängenden Graphen (Trunk ↔ MTP-Head, Producer ↔ Consumer) muss die Auflösung **beide Seiten gemeinsam** betrachten, nicht den Konfliktblock isoliert. Solche Bugs überleben Build + `llama-bench` und brauchen ein funktionales Gate (MTP-Accept-Rate).

**Noch zu tun:** MTP-Draft auf dem Server funktional gegentesten (Accept-Rate mit einem qwen35-MTP-Modell, Layer-Split), um den Fix zu bestätigen.

---

## Konflikt-Übersicht (9 Dateien)

| Datei | Art des Konflikts | Risiko |
|---|---|---|
| `ggml/src/ggml-hip/CMakeLists.txt` | mechanisch | niedrig |
| `src/llama-cparams.h` | Rename + Feld-Ergänzung | niedrig |
| `src/llama-context.h` | Rename + Feld-Ergänzung | niedrig |
| `src/llama-context.cpp` | Rename + Logik (3 Stellen) | mittel |
| `src/llama-graph.cpp` | Kommentar-only | niedrig |
| `src/models/qwen35.cpp` | Strukturkonflikt (Norm/Gather-Reihenfolge) | **hoch** |
| `src/models/qwen35moe.cpp` | identisch zu qwen35.cpp | **hoch** |
| `common/speculative.cpp` | tiefe Feature-Verschmelzung (6 Blöcke) | **hoch** |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 6 Blöcke, 4 mechanisch + 2 Feature-Merges | mittel |

---

## Details je Datei

### 1. `ggml/src/ggml-hip/CMakeLists.txt`
Upstream ergänzt `-fno-finite-math-only` neben eurem `-ffast-math`-Flag (aus Commit `eebef77cf`).
**Entscheidung:** upstream übernommen — schränkt `-ffast-math` gezielt ein (keine NaN/Inf-Annahme, wichtig für Softmax/Attention mit `-inf`-Masken), behält aber den restlichen Speedup.

### 2. `src/llama-cparams.h`
Upstream benennt `embeddings_pre_norm(_masked)` → `embeddings_nextn(_masked)` um (reines Rename). Euer Fork hatte zusätzlich `mtp_prefill_kv_only` ergänzt.
**Entscheidung:** neue Namen übernommen, `mtp_prefill_kv_only` behalten.

### 3. `src/llama-context.h`
Gleiches Rename-Muster für die Methoden (`set_embeddings_pre_norm` → `set_embeddings_nextn`). Upstream ergänzt neu `set_embeddings_layer_inp` und `set_nextn_layer_offset`. Euer Fork hat zusätzlich `set_mtp_prefill_kv_only` sowie die `pre_norm_accum`-Methoden (deferred-prefill-Puffer).
**Entscheidung:** alle vier Methoden-Gruppen kombiniert (upstream-Namen + Fork-Erweiterungen).

### 4. `src/llama-context.cpp` (3 Konfliktstellen)
- **Pipeline-Parallel-Gate (Zeile ~404):** Euer Fork hatte die `n_devices() > 1`-Bedingung entfernt und `n_layer_all` durch `n_layer` ersetzt (Teil des tensor-parallel-Fixes aus Commit `6699c5a14`, der Meta-Devices mit `n_devices()==1` unterstützt). Die Device-Prüfung ist weiter unten in der bestehenden OR-Klausel (`LAYER && n_devices>1) || TENSOR`) abgedeckt.
  **Entscheidung:** HEAD-Version behalten (bewusste, aktuelle Fork-Änderung).
- **Hidden-State-Extraktion (Zeile ~2017):** Upstream benennt `n_embd = hparams.n_embd` → `hparams.n_embd_out()` und `embd_pre_norm`/`t_h_pre_norm` → `embd_nextn`/`t_h_nextn` um. Euer Fork hat zusätzlich den `accum_active`-Zweig für das deferred-prefill-Feature (async D2H direkt in einen positions-indizierten Pinned-Buffer, mit Ring-Drain-Logik zur Vermeidung von Buffer-Korruption bei Pipeline-Parallelismus).
  **Entscheidung:** upstream-Naming übernommen, Fork-Logik (`accum_active`-Zweig) vollständig erhalten.
- **C-API-Funktionen (Zeile ~3776):** `llama_get_embeddings_pre_norm()` rief eine inzwischen nicht mehr existierende Methode `ctx->get_embeddings_pre_norm()` auf (durch das Rename in `llama-context.h` entfernt) — **wäre ein Compile-Fehler gewesen**. Upstream ergänzt neu `llama_set_embeddings_layer_inp`.
  **Entscheidung:** die verwaiste `llama_get_embeddings_pre_norm`-Funktion entfernt (stale API, nirgends mehr deklariert), Fork-spezifische Funktionen (`llama_set_mtp_prefill_kv_only`, `llama_*_embeddings_pre_norm_accum`) sowie upstreams neue `llama_set_embeddings_layer_inp` behalten.

### 5. `src/llama-graph.cpp`
Reiner Kommentarkonflikt — beide Seiten erklären denselben Sachverhalt (KQ-Mask bleibt unallokiert, wenn der Graph nur K/V ohne Attention schreibt), jeweils mit unterschiedlichem Beispiel (DFlash vs. MTP-Phase-2b).
**Entscheidung:** beide Beispiele in einem Kommentar zusammengeführt.

### 6./7. `src/models/qwen35.cpp` und `src/models/qwen35moe.cpp` (identisches Muster)
**Kernkonflikt:** unterschiedliche Reihenfolge von Norm und Gather bei der Extraktion des MTP-Hidden-States:
- **HEAD:** sichert `t_h_pre_norm` **vor** der finalen Norm, für alle Positionen (ungegatert); Norm + Gather laufen danach nur über die `n_outputs`-Zeilen, geschützt durch `if (n_outputs > 0)`.
- **upstream:** wendet die Norm zuerst auf alle Positionen an, sichert danach `t_h_nextn` (nachnormiert, alle Positionen), gathert erst danach für die Logit-Berechnung. Kein `n_outputs > 0`-Guard.

Der Doku-Kommentar in `llama-graph.h:836` ("hidden state **before** final output norm") passt zur HEAD-Reihenfolge, nicht zu upstream — ein Hinweis auf möglichen Upstream-Doku-Drift oder eine bewusste Semantikänderung dort.

**Diese Entscheidung wurde explizit mit dem Nutzer abgestimmt** (nicht automatisch getroffen), da HEADs Deferred-Prefill-Optimierung in `speculative.cpp`/`llama-context.cpp` exakt auf der Pre-Norm-Semantik zwischen Draft- und Target-Context beruht.
**Entscheidung:** HEAD-Struktur (Pre-Norm-Extraktion vor Norm/Gather, `n_outputs>0`-Guard) beibehalten, nur auf `nextn`-Namensgebung umgestellt.
**Restrisiko:** Falls upstream die Post-Norm-Reihenfolge aus einem funktionalen Grund (nicht nur Stil) geändert hat, könnte das bei zukünftigen non-MTP-Modellen zu Abweichungen führen — sollte bei nächstem Sync erneut geprüft werden.

### 8. `common/speculative.cpp` — komplexester Konflikt (6 Blöcke)
Zwei unabhängig gewachsene Feature-Sets mussten kombiniert werden:
- **Fork:** `adaptive_disable` (MTP pro Sequenz deaktivieren bei nicht-finiten Draft-Wahrscheinlichkeiten) und `deferred_prefill` (Prompt-Hidden-States werden ohne Pro-Chunk-Sync in einen Pinned-Buffer akkumuliert und erst bei `begin()` in einem Rutsch in den Draft-Context repliziert — behebt eine PP-Regression). Beides opt-in über `LLAMA_ENABLE_MTP_OPT`.
- **upstream:** `chain_heads` — Multi-Layer-MTP-Unterstützung (`n_mtp_layers > 1`), bei der jeder Kopf/Layer einen eigenen Decode-Durchlauf mit eigenem KV-Bereich braucht (`llama_set_nextn_layer_offset` je Head, `chain_h`-Pufferkette zur Weitergabe der Hidden-States zwischen Heads).

**Vorgehen:**
1. `process_decode()`-Signatur (bereits konfliktfrei gemergt) blieb bei HEADs `(tokens, positions, seqs, h_tgt_in)`-Parametern statt upstreams `batch_in`-Objekt — upstreams Codefragmente wurden entsprechend auf lokale Arrays umgeschrieben.
2. Der `chain_heads`-Decode-Loop (Multi-Head, `llama_memory_seq_rm` + `llama_set_nextn_layer_offset` je Head) wurde übernommen, aber so umgebaut, dass er weiterhin `h_tgt_in` (Fork) respektiert statt immer synchronisierend von `ctx_tgt` zu lesen.
3. **Scope-Bug gefunden und gefixt:** `h_tgt` war im ursprünglichen HEAD-Konfliktzustand nur innerhalb des `if (!is_mem_shared)`-Blocks deklariert, wurde aber danach (in der `verify_h`-Extraktionsschleife) weiter benutzt — hätte nicht kompiliert bzw. bei is_mem_shared-Pfaden falsch gebunden. Deklaration vor den `if`-Block gezogen.
4. Finale Extraktion (`verify_h`): HEADs zeigerbasierter Zugriff (`h_tgt + offset`) beibehalten statt upstreams `llama_get_embeddings_nextn_ith()` — notwendig, damit der `h_tgt_in`-Override (deferred prefill) korrekt greift; die `_ith`-Variante würde immer aus dem regulären `ctx_tgt`-Buffer lesen und den deferred-prefill-Pfad brechen.
5. Im `draft()`-Loop: HEAD enthielt eine ältere, nicht mehr passende Direkt-Decode-Sequenz (`if (batch.n_tokens==0) return; llama_decode(...)`), die mit der bereits (konfliktfrei) gemergten `chain_heads`-`while`-Schleife weiter unten kollidiert hätte (Doppel-Decode / falsche Kontrollstruktur). Stattdessen wurde upstreams `chain_h[seq_id]`-Seeding übernommen, das von der bestehenden `while`-Schleife benötigt wird. Der Leerlauffall (`n_drafting == 0`) ist durch die `while (n_drafting > 0)`-Bedingung bereits sicher abgedeckt.
6. Alle verbleibenden Aufrufe der (jetzt nicht mehr deklarierten) `llama_set/get_embeddings_pre_norm()`-Funktionen auf `llama_set/get_embeddings_nextn()` umgestellt (sonst Compile-Fehler).

**Restrisiko:** Das ist der am wenigsten mechanisch verifizierte Teil des Merges. Empfehlung: gezielter Funktionstest mit `LLAMA_ENABLE_MTP_OPT` gesetzt **und** ungesetzt, für ein Modell mit `n_mtp_layers > 1` (chain_heads-Pfad) sowie eines mit `n_mtp_layers == 1`.

### 9. `ggml/src/ggml-cuda/ggml-cuda.cu` (6 Blöcke)
- **4 Blöcke rein mechanisch:** jeweils eine Konfliktseite leer (reine Fork- bzw. reine Upstream-Ergänzung ohne Überlappung, u. a. ~270 Zeilen NCCL/Tensor-Parallel-Kommunikationscode des Forks). Automatisiert per Skript aufgelöst und stichprobenartig verifiziert.
- **`ggml_cuda_mul_mat()` (Zeile ~2853):** Fork ergänzt einen Fast-Path für GCN-repackte Gewichte (eigener Matvec/Dequant-Pfad, siehe `repack-gcn.cu`); upstream ergänzt `GGML_TENSOR_BINARY_OP_LOCALS` (liefert u. a. `ne11`, das weiter unten in der Funktion gebraucht wird — ohne dieses Makro hätte der Code nicht kompiliert) sowie einen Hadamard-Fusion-Hint-Shortcut. Eine lokale `split`-Variable aus dem Fork wird im gesamten Funktionskörper nirgends referenziert und wurde als toter Code entfernt.
  **Entscheidung:** GCN-Repack-Check zuerst, danach `GGML_TENSOR_BINARY_OP_LOCALS` + Hadamard-Check kombiniert.
- **`ggml_backend_cuda_device_supports_buft()` (Zeile ~6430):** upstream vereinfacht die Buffer-Typ-Prüfung auf `ggml_backend_buft_is_cuda`, entfernt aber `ggml_backend_buft_is_cuda_split`/`ggml_backend_buft_is_cuda_repack` — beides Fork-exklusive Buffer-Typen (tensor-split bzw. GCN-repack), die andernorts im Fork aktiv genutzt werden.
  **Entscheidung:** HEAD-Version (mit `split`/`repack`-Support) behalten, da sonst diese Buffer-Typen als "nicht unterstützt" abgelehnt würden.

---

## Offene Punkte / Empfehlungen vor Merge in `master`

1. **Build:** CUDA- und HIP-Build durchführen (beide Backends waren an Konflikten beteiligt).
2. **MTP-Rauchtest:** Spekulativdekodierung mit einem MTP-fähigen Modell (Qwen3.5/Qwen3.5-MoE) testen — mit und ohne `LLAMA_ENABLE_MTP_OPT`, idealerweise mit einem `n_mtp_layers > 1`-Modell für den `chain_heads`-Pfad.
3. **Tensor-Parallel-Rauchtest:** Da `llama-context.cpp` (Pipeline-Parallel-Gate) und `ggml-cuda.cu` (`split`/`repack`-Buffer-Support) beide tensor-parallel-relevant sind, den `-sm tensor`/`-tps`-Pfad gegenprüfen.
4. **qwen35(moe).cpp-Semantik** beim nächsten Upstream-Sync erneut prüfen — falls upstream die Post-Norm-Reihenfolge dort aus funktionalen (nicht nur stilistischen) Gründen geändert hat, könnte das relevant werden, sobald der Fork versucht, diesen Pfad noch enger an upstream zu führen.
5. Kein automatisierter Test wurde in dieser Session ausgeführt (auf Nutzerwunsch) — die obigen Punkte sind manuell nachzuholen.
