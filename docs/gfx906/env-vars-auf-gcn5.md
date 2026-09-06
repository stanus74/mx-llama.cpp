# Welche Umgebungsvariablen auf `gcn5` noch etwas tun

**Stand:** 2026-08-05

Die `FEATURES.md` des Original-Forks (`mxxm-t/mx-llama.cpp`) empfiehlt vier Exports. Alle vier
lesen **fork-exklusiven Code**, den der `gcn5`-Branch nicht trägt.

| Export | Gelesen von | auf `gcn5` |
|---|---|---|
| `GGML_ENABLE_CUSTOM_AR=1` | `ggml-cuda.cu:1603` (Fork) | **wirkungslos** |
| `HSA_FORCE_FINE_GRAIN_PCIE=1` | `tp-allreduce.cu:656` (Fork) | **wirkungslos, potenziell schädlich** |
| `LLAMA_ENABLE_MTP_OPT=1` | `common/speculative.cpp:547` (Fork) | **wirkungslos** |
| `GPU_MAX_HW_QUEUES=8` | ROCm-Runtime (wirkt immer) | **gemessen: kein Effekt** |

## `HSA_FORCE_FINE_GRAIN_PCIE` widerspricht der eigenen Konfiguration

`~/.bashrc` auf `x99` schaltet P2P bewusst ab (`HSA_P2P_DISABLE=1`, `HIP_FORCE_P2P_DISABLE=1`,
`HSA_FORCE_P2P=0`), und `scripts/mx-compile.sh` begründet das: auf dem X99-Board gibt es keinen
PLX-Switch zwischen den MI50-Root-Ports, P2P ist nicht verfügbar, Custom-AR fällt ohnehin auf RCCL
zurück. Fine-Grain-Speicher zu erzwingen bringt hier keinen schnelleren Pfad, nur teurere
Allokationen. **Nicht setzen.**

## `GPU_MAX_HW_QUEUES` — gemessen, kein Effekt

Empfohlen war er für MoE-Durchsatz unter `-tps`, einem Feature, das `gcn5` nicht hat. Gemessen mit
korrekter Umgebung (`HSA_XNACK=0` usw.), `-r 2`:

**Ornith-1.0-35B (MoE), eine GPU:**

| Wert | pp2048 | tg128 |
|---|---:|---:|
| unset | 764,87 ± 1,57 | 56,60 ± 0,48 |
| 4 | 763,70 ± 1,13 | 55,61 ± 0,43 |
| 8 | 765,20 ± 1,78 | 56,53 ± 0,55 |

**Qwen3.6-27B, zwei GPUs, `-sm tensor`, `-ctk/-ctv q8_0`:**

| Wert | pp2048 | tg128 |
|---|---:|---:|
| unset | 389,18 ± 0,59 | 23,30 ± 1,17 |
| 4 | 389,83 ± 0,46 | 23,12 ± 1,34 |
| 8 | 390,30 ± 0,32 | 23,23 ± 1,27 |

Beides flach. Die Fork-Kommentare warnen zudem selbst vor Nebenwirkungen
(`dst->main is not reliable on HIP under GPU_MAX_HW_QUEUES=8`). **Nicht setzen.**

## Was stattdessen gilt

Nötig und belegt (steht bereits in `~/.bashrc`):

```bash
export HSA_XNACK=0    # ohne dies melden sich die Karten als xnack+ und MoE-Prefill stürzt ab
```

Siehe [hipblas-sgemm-moe-router-crash.md](hipblas-sgemm-moe-router-crash.md).

## Kernel: `amdgpu.noretry` nicht auf 0 zwingen (2026-09-06)

`HSA_XNACK=0` hat einen kernelseitigen Gegenpart: `amdgpu.noretry`. Der Modulstandard ist **`-1`
(auto)**; `noretry=0` erzwingt **Retry-Page-Faults**, also genau den Mechanismus, den XNACK nutzt.

Auf `x99` stand `amdgpu.noretry=0` in der Kernel-Kommandozeile — im Widerspruch zu `HSA_XNACK=0`
im Userspace. Symptom: ein `workqueue: interrupt_wq [amdgpu] hogged CPU for >10000us`-Sturm mit
exponentiell wachsender Zählung (4 → 8 → … → 2048 in vier Minuten), dabei eine Anfrage, die
1h47m in der Warteschlange hing und dann mit `no valid JSON data found in stream` scheiterte.
Kein GPU-Reset, kein OOM, kein Panic.

**Behoben durch Streichen von `amdgpu.noretry=0`** aus `/etc/default/grub` (+ `update-grub`,
Reboot). Danach `noretry = -1`, Karten weiterhin `gfx906:sramecc+:xnack-`, keine Regression:
`Qwen3.8-27B-UD-Q6_K_XL` pp4096 383,73 ± 0,74 gegen 380,44 ± 1,05 vorher. Der Kernel wechselte
im selben Zug von 6.8.0-138 auf -139, die kleine Verbesserung ist deshalb nicht zuschreibbar.

**Regel:** Wer `HSA_XNACK=0` setzt, sollte `amdgpu.noretry` auf `auto` lassen. Beides gegenläufig
zu konfigurieren heißt, dass die GPU Faults nimmt, mit denen der Userspace nicht rechnet.

Offen: `HSA_ENABLE_SDMA=0` in `hip_env_dual` schaltet die DMA-Engines ab, Kopien laufen dann als
Blit-Kernel über die Compute-Queues — mehr Completions, mehr Interrupts. Stammt vermutlich aus
derselben Debug-Phase wie die P2P-Flags und ist ungemessen.

---

Weiterhin richtig, unabhängig vom Fork:

- **`-lm dio`** — Upstream-Flag, mmap hängt auf diesem Stack.
- **Multi-GPU über `-sm tensor`**, ohne `-tps` — das Flag existiert auf `gcn5` nicht, die
  `-tps`-Beispielkommandos der `FEATURES.md` laufen hier nicht.
