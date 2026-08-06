# Multi-GPU: `-sm tensor` lohnt sich hier, trotz fehlendem P2P

**Gemessen:** 2026-08-05 · MI50 16 GB + MI50/Radeon-VII 32 GB, X99-Board, ROCm 6.3.4

`Qwen3.6-27B-Fable-…-Q6_K` (22,37 GiB), `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -r 2 -p 2048 -n 128 -lm dio`,
korrekte Umgebung (`HSA_XNACK=0` usw.):

| Konfiguration | pp2048 | tg128 |
|---|---:|---:|
| 1 GPU (32 GB) | 214,29 ± 0,90 | 15,43 ± 0,03 |
| 2 GPUs, `-sm layer` | 290,43 ± 0,50 | 16,31 ± 0,04 |
| **2 GPUs, `-sm tensor`** | **389,51 ± 0,47** | **23,25 ± 1,24** |

**`-sm tensor` ist die richtige Wahl** — +82 % Prefill und +51 % Decode gegenüber einer Karte.

## Warum, obwohl P2P abgeschaltet ist

Naheliegende Vermutung war das Gegenteil: auf diesem X99-Board gibt es keinen PLX-Switch zwischen
den MI50-Root-Ports, P2P ist nicht verfügbar und in `~/.bashrc` zusätzlich hart abgeschaltet, also
läuft jede AllReduce über Host-Speicher. Das Modell passt zudem auf die 32-GB-Karte allein — es
*muss* gar nicht geteilt werden.

Trotzdem gewinnt der Split deutlich, weil **Token-Generierung speicherbandbreitenbegrenzt** ist.
Zwei Karten liefern doppelte Bandbreite für die Gewichte, und dieser Gewinn übersteigt die
AllReduce-Kosten über PCIe. Layer-Split bringt dagegen wenig (+6 % tg), weil dort immer nur eine
Karte gleichzeitig arbeitet — er löst ein Kapazitäts-, kein Bandbreitenproblem.

## Nicht mit MoE verwechseln

Ein A3B-MoE auf **einer** Karte (Ornith-35B: tg128 56,6) ist schneller als dieses dichte 27B auf
**zwei** (23,3). Das ist ein Architektur-, kein Topologie-Effekt: pro Token rechnet ein A3B nur mit
rund 3B aktiven Parametern. Die beiden Größen nicht gegeneinander lesen.

## Konsequenz für den GCN5-Plan

Schritt 5 (Serverumstellung) kann `-sm tensor` unverändert übernehmen. Der Wegfall von `-tps`
kostet nichts, solange nur zwei Karten im Spiel sind — Multi-Stage-Pipelining braucht mehr GPUs,
als hier stecken.
