# gfx906 / Dual-MI50-MI60 Optimierungsguide

Referenzdokument für zukünftige Chats. Fasst Hardware, Software-Entscheidungen, angewendete
Patches, Konfiguration und offene Punkte zusammen. Alles hier Beschriebene wurde real auf dem
System getestet (Build, Bench, `test-backend-ops`), nicht nur aus Dokumenten übernommen —
Ausnahmen sind explizit markiert.

---

## 1. Hardware

| | |
|---|---|
| Board | Asus X99-WS (Intel C610-Chipsatz, Xeon E5-2690 v4) |
| GPU 0 | AMD Instinct MI50/MI60, **16 GB** VRAM |
| GPU 1 | AMD Radeon Graphics (MI50/MI60-Familie), **32 GB** VRAM |
| Architektur | gfx906 (Vega20/CDNA1, **keine Matrix-Cores** — die gibt's erst ab CDNA/gfx908+) |
| ROCm | 6.3.4 |
| PCIe-Topologie | Beide Karten an **getrennten Root-Ports** der CPU (`00:02.0`/`00:03.0`), **kein PLX-Switch** dazwischen → **P2P nicht verfügbar**, bestätigt via `lspci -tv` |

**Konsequenz aus der asymmetrischen VRAM-Verteilung + fehlendem P2P:**
- Custom-AllReduce (`GGML_ENABLE_CUSTOM_AR`) fällt automatisch auf RCCL zurück, bringt nichts
- Layer-Split mit `--fit`/TP muss die ungleiche VRAM-Größe berücksichtigen (OOM-Risiko auf der 16-GB-Karte)

---

## 2. Software-Stack — welcher Fork wofür

| Repo | Rolle | Verifiziert? |
|---|---|---|
| `ggml-org/llama.cpp` | Echtes Mainline | ✅ Immer aktuell halten als Referenz |
| `mxxm-t/mx-llama.cpp` | Fork: Multi-Stage-TP (`-tps`), Custom-AllReduce, MTP-Optimierungen, inline gfx906-Tuning | ✅ Geklont, Divergenz/Konflikte real getestet |
| `iacopPBK/llama.cpp-gfx906` | Ursprungs-Fork der gfx906-Kernel (`gfx906/`-Ordner: DPP-Reductions, FlashAttention-Tiles, MMVQ) | ✅ Real existent, aktiv gepflegt |
| `DENEB1312/mx-llama.cpp` | **Aktuell in Benutzung** (`/opt/mx-llama.cpp`): mx-llama.cpp + iacopPBK-Kernel sauber integriert, 4 Commits vor mxxm-t/master | ✅ Geklont, gebaut, gebencht |
| `skyne98/llama.cpp-gfx906` | Fork-of-fork von iacopPBK, Branch `ruquant-w4a4` enthält den Kernel-Port (nicht der `gfx906`-Branch!) | ✅ Real, aber 1208 Commits hinter Upstream |

### ⚠️ Als Fake/nicht-existent identifiziert — NICHT verwenden
| Repo/Verweis | Befund |
|---|---|
| `AmesianX/TurboQuant` | README behauptet Archivierung wegen gescheitertem vLLM-Port; extrem detaillierte, aber höchstwahrscheinlich fabrizierte Versionshistorie (v1.0–v1.9), erwähnt "Opus 4.8"/"Fable 5" im Text — starkes Fake-Signal |
| `stanus74/llama-mi50.cpp` | **404 — existiert nicht.** Tauchte als Quellenverweis in einem "Optimierungsprotokoll" auf, das plausibel klang, aber auf einen toten Link zeigte |
| `arte-fact/llamacpp-gfx-906-turbo` | Real existent, aber Fokus auf TurboQuant-KV-Kompression (DGX-Spark/CUDA-lastig), für dieses Setup nicht relevant |
| `Llaminar/llaminar` | Eigenständige C++-Engine, kein llama.cpp-Fork, Alpha-Status, keine Gemma4/SmolLM3/Kimi-Unterstützung — nur für Mixed-CUDA+ROCm-Experimente interessant |
| `eslowney/llama.cpp-gfx906` | Real, aber nur für Head-Dim=128-Modelle (Qwen3-30B-Klasse), zu eng für die genutzte Modell-Mischung |

**Lektion:** Vor jedem Vertrauen in ein "Repo-Protokoll" oder Diff-Dokument den referenzierten
Link/Fork tatsächlich klonen oder per `web_fetch` prüfen (404 = Warnsignal), nicht nur den
Inhalt auf Plausibilität bewerten.

---

## 3. Build — Erkenntnisse & finales Skript

### Kritische Build-Flags (verifiziert per 5-Punkt-Bisect)

| Flag | Wert | Warum |
|---|---|---|
| `GGML_HIP_RCCL` | `ON` | **Pflicht.** Ohne RCCL: ~720 statt ~1000 t/s pp512 bei TP — kein Code-Bug, reine Build-Flag-Frage, per Bisect über 5 Commits bestätigt |
| `GGML_HIP_ROCWMMA_FATTN` | `OFF` | gfx906 hat keine Matrix-Cores; rocWMMA unterstützt nur gfx908+/gfx11xx+/gfx12xx — Paket kann installiert sein, bringt aber nichts, evtl. sogar Crash-Risiko |
| `CMAKE_HIP_FLAGS` | `-ffast-math` | Isoliert aus DENEB1312-Fork übernommen, ohne vollen Merge. Leichte FP-Genauigkeitseinbuße möglich, praktisch vernachlässigbar bei quantisierten Modellen |
| `GGML_LTO` | `ON` (Produktiv-Build) | Für Bisect/Debugging testweise `OFF` — Build-Zeit sinkt deutlich, Performance-Effekt selbst nicht gemessen relevant |
| `GGML_HIP_NO_VMM` | `ON` | Aus ursprünglichem Skript übernommen, unverändert |

### Finales Build-Skript: `mx-compile.sh`
Enthält: Ninja-Erkennung (Fallback Make), ccache, alle obigen Flags, `FASTMATH=0`-Toggle zum
Abschalten von `-ffast-math`, informativer Abschluss mit Start-Befehlen je Anwendungsfall.
→ Separat als Datei vorhanden, bei Bedarf neu erzeugen lassen.

### Bekannte Stolperfallen beim Bauen
- **Verzeichnisverwechslung:** mehrere Checkouts (`/opt/llama.cpp` Mainline, `/opt/mx-llama.cpp`
  Fork, versehentlich entstandene Kopien wie `mx-llama-org.cpp`) können leicht durcheinander
  geraten. Vor jedem Bench/Build **immer** `pwd` + `git log -1 --format="%H %s"` gegenchecken.
- **Stale Build ≠ Git-Stand:** `git log` kann einen Commit zeigen, während das kompilierte
  Binary (`strings build/bin/... | grep <erwarteter-String>`) einen älteren Stand widerspiegelt,
  wenn zwischen letztem Build und letztem `git pull`/`checkout` nicht neu gebaut wurde. Prüfen
  mit `strings <binary> | grep -c <marker>` bzw. bei Shared-Libs alle `.so`-Dateien durchsuchen
  (`llama-cli` ist nur ein dünner Wrapper um `libllama-cli-impl.so`/`libllama.so`).

---

## 4. Angewendete Kernel-Patches

### 4.1 `nwarps`-Tuning für MMQ (Prompt-Processing-Boost)

**Hintergrund:** llama.cpp wählt Warps/Block über eine generische `256/warp_size`-Heuristik,
kalibriert für Matrix-Core-Architekturen. Auf **gfx906 ohne Matrix-Cores** unterdimensioniert
das die Compute-Units deutlich. Ursprungsquelle: llama.cpp-Discussion
[#23881](https://github.com/ggml-org/llama.cpp/discussions/23881) — bis zu +76 % pp512 bei
Q8_0 (MI50 32GB, andere Quelle).

**Zentrale Randbedingung:** Host- (`mmq_get_nwarps_host`) und Device-Funktion
(`mmq_get_nwarps_device`) **müssen synchron geändert werden**. Ein Patch nur der Host-Seite
führte real zu `ROCm error: unspecified launch failure` (reproduziert).

**Umsetzung** (Datei `ggml/src/ggml-cuda/mmq.cuh`, Mainline-Version ohne `type`-Template):
Macro `GGML_MMQ_NWARPS_GFX906` (Default `8`), Host-Zweig `cc == GGML_CUDA_CC_VEGA20` und
Device-Zweig `defined(__gfx906__)` nutzen dasselbe Macro.

**Gemessenes Ergebnis** (`/opt/llama.cpp`, Commit `e920c523e`, Qwopus3.6-27B-Coder Q6_K,
single-GPU):

| nwarps | pp512 | Δ |
|---|---|---|
| 4 (alt) | ~163 t/s | Baseline |
| **8 (neu)** | **190.99 ± 0.57 t/s** | **+17.3 %** |

**Korrektheit:** `test-backend-ops -o MUL_MAT` mehrfach sauber (0 Fails). Ursprüngliche
Discussion nannte ein seltenes, nicht-deterministisches Fehlverhalten bei `q5_1, m=16, n=1,
k=32` bei nwarps=8 — bei diesem Setup nicht beobachtet, aber nicht kategorisch ausschließbar.

**⚠️ Bekannte Grenze:** Ein pauschaler Versuch, nwarps=8 für **alle** Quant-Typen zu erzwingen
(nicht nur Q5_K/Q6_K, sondern auch Q4_0), führte zu `ROCm error: unspecified launch failure`
bei `Q4_0, m=16, n=8, k=256` — reproduzierbar. Nicht ohne erneuten Korrektheitstest auf weitere
Quant-Typen ausweiten. Q4_0 wurde mit dem finalen (synchronisierten) Patch nicht erneut
verifiziert.

**Wiederherstellungs-Skript nach `git pull`:** `llama-pull-repatch.sh` — idempotent, bricht
kontrolliert ab statt blind zu patchen, falls Upstream die Funktionsstruktur ändert.

### 4.2 Noch nicht umgesetzt, real verifiziert als sinnvoll

- **DPP-basierte Warp-Reduktionen** (`common.cuh`) — Quelle: echter `iacopPBK/llama.cpp-gfx906`-
  Fork. Unified-Shuffle-Dispatch, DPP-Intrinsics statt LDS-Roundtrip auf gfx906.
- **FlashAttention GCN-Tuning** (`fattn-common.cuh`, `fattn.cu`) — Q8-optimierte
  Tile-Kernel-Auswahl, ebenfalls iacopPBK.
- **`repack-gcn`** — bereits Teil des aktuell genutzten DENEB1312-Forks (`/opt/mx-llama.cpp`),
  Q4_K-Weight-Repacking für Wave64-Alignment, gemessen 470→740 GB/s Bandbreite in der
  Fork-eigenen Doku. Aktivierung/Default-Status (`GGML_CUDA_REPACK=1`?) noch nicht geprüft.
- **Vulkan-Backend MI50-Optimierung** — laut PyPI-Changelog `llama-cpp-pydist` existiert
  Mainline-PR #22933 ("vulkan: opt mul_mat_vecq for mi50"), aktiviert Subgroup-Arithmetik für
  eine Allowlist an AMD-GPUs. Kein Fork nötig, aber Vulkan- statt HIP-Backend wäre ein
  Architekturwechsel — nicht getestet.
- **Speichertakt-Overclocking via `upp`** (`MEM_MAX`) — laut mixa3607-Referenzwerten der
  wirkungsvollste Hebel für PP/TG (HBM2-bandbreitengebunden), aber invasiver als bisherige
  Änderungen. Nur Power-Limit + TdcLimitGfx wurden bisher umgesetzt, nicht der Speichertakt.

---

## 5. Bekannte Inkompatibilitäten (real reproduziert, nicht nur vermutet)

### 5.1 Tensor-Parallel (`-sm tensor`) + MTP-Speculative-Decoding
**Crasht zuverlässig** mit `ROCm error: out of memory` auf der kleineren 16-GB-Karte.
Ursache: TP erzwingt CPU-Sampler-Fallback für den MTP-Draft (`backend sampling not supported
with SPLIT_MODE_TENSOR`), was zusätzlichen Speicherdruck erzeugt, der die 16-GB-Karte sprengt.

**Konsequenz:** MTP-Modelle → **Layer-Split**, nicht TP. TP nur für reine PP-lastige
Workloads ohne Speculative Decoding.

### 5.2 `--fit on` + MTP ohne explizites `-c`
Ohne gesetzte Kontextgröße läuft `-c` auf `n_ctx_train` (z. B. 262144) hoch; zusammen mit dem
MTP-Draft-Kontext sprengt das die 16-GB-Karte. **Immer `-c` explizit setzen**, besonders bei
MTP-Modellen auf der asymmetrischen Kartenkombination.

### 5.3 P2P generell nicht verfügbar
Siehe Abschnitt 1 — PCIe-Topologie ohne PLX-Switch. `HSA_P2P_DISABLE=1`,
`HIP_FORCE_P2P_DISABLE=1`, `HSA_FORCE_P2P=0` sind bewusst in `.bashrc` gesetzt (Grund:
"X99 PCIe-Bus-Hänger" — historisches Stabilitätsproblem). Nicht rückgängig machen ohne
Reboot-Zugang griffbereit.

---

## 6. Power-/Thermal-Tuning (aktiv, via `upp` + systemd)

Tool: [`upp`](https://github.com/sibradzic/upp) (SoftPowerPlay-Table-Editor), installiert via
pipx (`/home/pat/.local/bin/upp`).

**Aktive Konfiguration** (Service: `/etc/systemd/system/mi50-tdclimit.service`, enabled):

```bash
ExecStart=/bin/bash -c 'for d in /sys/class/drm/card*/device/pp_table; do \
  /home/pat/.local/bin/upp -p "$d" set --write \
  smcPPTable/TdcLimitGfx=150 \
  smcPPTable/SocketPowerLimitAc0=180; \
done'
```

| Parameter | Vorher (Werk) | Nachher | Zweck |
|---|---|---|---|
| `TdcLimitGfx` | 330 | 150 | Hotspot-Stromspitzen begrenzen (~8-12°C niedrigere Junction-Temp laut Referenzwerten) |
| `SocketPowerLimitAc0` | 250 (Card0) / 225 (Card1) | 180 | Gesamt-Package-Power-Cap, sekundär auch temperatursenkend |

Backup der Original-PP-Tabellen vor Änderung: `/root/pp_table_card{0,1}_backup_<datum>.txt`
(per `upp ... dump`).

**Wichtiger Unterschied der beiden Parameter:** `TdcLimitGfx` kappt lokale Stromspitzen
(Hotspot-fokussiert), `SocketPowerLimitAc0` senkt die kontinuierliche Gesamtleistung
(wirkt auf alle Temperatursensoren). Komplementär, nicht redundant.

---

## 7. llama-swap-Konfiguration — Lektionen

- **Separates Macro für TP-Modelle** (`hip_env_tp`, `default_args_tp` ohne `--fit`, mit
  `--split-mode tensor`) getrennt von Standard-Layer-Split-Modellen halten.
- **MTP-Modelle NIEMALS auf TP-Macro umstellen** (siehe 5.1).
- Bei Umstellung mehrerer Modelle auf einen neuen Server-Binary-Pfad (`server`-Macro): **jedes
  Modell einzeln durchtesten**, nicht pauschal vertrauen — ein Modell ohne explizites `-c` kann
  mit neuem Backend anders crashen als vorher (siehe 5.2, real passiert bei `qwen36-27b`).
- Fallback-Einträge (z. B. `qwen36-35b-layersplit` neben `qwen36-35b`) sind günstig, um bei
  Problemen ohne Config-Neuschrieb zurückschalten zu können.

---

## 8. Methodik-Hinweis für zukünftige Chats

Dieses Setup hat mehrfach gezeigt: **KI-generierte "Optimierungsprotokolle" und Fork-Vergleiche
können plausibel klingen und trotzdem auf erfundenen Repos/Zahlen basieren** (siehe Abschnitt 2,
Fake-Liste). Bevor ein Patch aus einem Dokument übernommen wird:

1. Referenzierten Fork/Link real klonen oder per Web-Fetch prüfen (404 = Warnsignal)
2. Bei Kernel-Änderungen: Host **und** Device-Seite auf Synchronität prüfen (siehe 4.1)
3. Nach jedem Patch: `test-backend-ops -o MUL_MAT` **mehrfach** (nicht-deterministische Fehler
   möglich), danach erst Performance-Vergleich
4. Bench-Ergebnisse mit Verzeichnis/Commit-Stand gegenchecken (`pwd`, `git log -1`) — leicht
   verwechselbar bei mehreren Checkouts
5. Pauschalisierungen (ein Fix für alle Quant-Typen/alle Architekturen) sind riskanter als
   selektive, getestete Änderungen — siehe Q4_0-Crash in 4.1