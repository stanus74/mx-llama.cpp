# AGENTS.md — Arbeitsanweisungen für KI-Assistenten (mx-llama.cpp)

Dies ist ein **privater Fork** von `llama.cpp`, fokussiert auf **AMD/ROCm (gfx906 / MI50)**.
Die Upstream-Contributor-/Anti-AI-PR-Policy gilt hier **nicht** (private Forks sind davon ausgenommen).

## Sprache & Kommunikation

- **Kommunikation auf Deutsch.** Code, Kommentare, Commit-Messages und technische Doku auf Englisch.
- Direkt und konkret. Bei Konfliktauflösungen: kurz begründen, welche Seite warum gewählt wurde.
- Bei echten Semantik-Entscheidungen (nicht mechanisch) **vor** der Umsetzung nachfragen, statt blind zu kombinieren.

## Fork-spezifische Features (nicht versehentlich wegmergen!)

Diese Erweiterungen existieren nur in diesem Fork und müssen bei Upstream-Merges erhalten bleiben:

- **MTP-Spekulativdekodierung** (`common/speculative.cpp`, `src/llama-context.*`): opt-in über
  `LLAMA_ENABLE_MTP_OPT`. Umfasst `deferred_prefill` (positions-indizierter Pinned-Accum-Buffer,
  `pre_norm_accum`), `adaptive_disable` (MTP pro Sequenz bei nicht-finiten Draft-Probs deaktivieren)
  und `mtp_prefill_kv_only` (Phase 2b, KV-only-Replay). Setzt Pre-Norm-Hidden-State-Semantik voraus.
- **Multi-Stage Tensor-Parallel + Custom AllReduce/NCCL** (`ggml-cuda.cu`, `llama-model.cpp`,
  `llama-context.cpp`): Meta-Device mit `n_devices()==1`, daher Pipeline-Gates über Modus statt
  `n_devices>1`; Layer→Stage-Mapping braucht `n_layer_all` (Gesamtzahl, nicht `n_layer()`).
- **GCN-repackte Gewichte** (`repack-gcn.cu`): eigener Matvec/Dequant-Pfad, `ggml_backend_buft_is_cuda_repack`.
- **Split-Buffer / Tensor-Split** (`ggml_backend_cuda_split_buffer_type*`, `-ts`): `ggml_backend_buft_is_cuda_split`.
- **gfx906-Kernel-Tuning** — Hardware-Details & Optimierungsregeln in
  [docs/gfx906-optimization-notes.md](docs/gfx906-optimization-notes.md) (ISA-Grenzen: kein MFMA,
  nur `v_dot4/8`/`dot2`; LDS-Bank-Padding; KV-Cache `HSD`; FP32-vs-QDQ-Abwägung; Latency-Hiding).

## Upstream-Merge-Workflow (wichtigste Lektion dieses Repos)

Große Upstream-Merges brechen fork-exklusiven Code oft **außerhalb** der von Git gemeldeten
Konfliktdateien, weil Git upstream-weite Umbenennungen/Löschungen automatisch (ohne Marker) anwendet,
aber fork-eigene Stellen ohne upstream-Gegenstück stehen lässt. Ablauf:

1. **Konflikte auflösen:** Fork-Semantik behalten, wo Fork-Features darauf beruhen; reine
   Upstream-Renames/-Refactors übernehmen. Bei struktureller Divergenz (nicht nur Rename) nachfragen.
2. **Nach jedem bemerkten Upstream-Rename SOFORT den GESAMTEN Baum greppen** (nicht nur Konfliktdateien)
   nach den alten Symbolnamen. Beispiele aus der Historie:
   - `embeddings_pre_norm` → `embeddings_nextn`
   - `t_h_pre_norm` → `t_h_nextn`
   - `hparams.n_layer` (Feld) → `n_layer_all` (Feld, Gesamt) **oder** `n_layer()` (Methode, ohne Nextn)
3. **Symbol-Diff gegen alten HEAD:** Funktionen/Structs, die im alten HEAD definiert waren, im Merge-
   Ergebnis fehlen, aber noch aufgerufen werden → stille Auto-Merge-Löschung (z. B. das komplette
   Split-Buffer-Subsystem wurde so entfernt und musste aus dem alten HEAD wiederhergestellt werden).
4. **Vollständige Merge-Reports** erstellen/pflegen (`MERGE_REPORT.md`): je Konfliktdatei Entscheidung +
   Begründung + Restrisiko, plus Nachträge für jeden Build-Fehler mit Ursache und Lehre.

## Build & Test

- **Zielplattform HIP / gfx906 (MI50).** Build läuft auf dem Server; erzeugt `build_mi50_*.log`.
- **Keine Tests/Builds ungefragt starten.** Wenn der Nutzer „kein test" sagt: nur committen/pushen.
- Kein lokaler Build hier vorausgesetzt — Verifikation erfolgt oft erst auf dem Server.

## Git-Konventionen

- **Nie direkt auf `master` committen** — auf einem Branch arbeiten (aktuell z. B. `merge-upstream-YYYYMMDD`).
- **Committen/Pushen nur auf Aufruf.** Aussagekräftige Commit-Messages: Was + Warum, bei Merge-Fixes
  auch, ob es ein Merge-Artefakt oder ein vorbestehender Fork-Bug war.
- Deaktivierte CI-Workflows per `.yml.disabled`-Konvention (nur self-hosted + HIP-relevante bleiben aktiv).

## CI / GitHub Actions

- Nur diese Workflows aktiv halten: `build-self-hosted`, `server-self-hosted`, `ui-build-self-hosted`,
  `ui-self-hosted`, `hip-quality-check`, `build-cache`. Alle generischen Upstream-Plattform-Workflows
  (CANN, IBM, RISC-V, Vulkan, SYCL, CUDA-Windows, Docker, Release, Python-Checks …) bleiben `.disabled`,
  damit keine unerwünschten Runs/Notifications entstehen.
