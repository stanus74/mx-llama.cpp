https://arkprojects.space/wiki/AMD_GFX906/perf-tuning

Perf tuning
Force PCIe speed
curl -L https://github.com/corundum/corundum/raw/refs/heads/master/fpga/lib/pcie/scripts/pcie_set_speed.sh > pcie_set_speed.sh
chmod +x pcie_set_speed.sh

# PCI bridge: Advanced Micro Devices, Inc. [AMD/ATI] Device 14a0 (rev 01)
AMDGPU_DEVICES=(
  '17:00.0'
  '1a:00.0'
  '31:00.0'
  '4b:00.0'
)
for (( i=0; i<${#AMDGPU_DEVICES[@]}; i++ )); do
  sudo ./pcie_set_speed.sh "${AMDGPU_DEVICES[$i]}" 4
done


Overclock / PowerLimit
upp - https://github.com/sibradzic/upp

# PP parameters
echo '
SmallPowerLimit1: $TDP_MAX
SmallPowerLimit2: $TDP_MAX
BoostPowerLimit: $TDP_MAX
PowerSavingClockTable:
  PowerSavingClockMax:
    PowerSavingClockMax 0: $GPU_MAX
smcPPTable:
  SocketPowerLimitAc0: $TDP_MAX
  SocketPowerLimitDc: $TDP_MAX
  TdcLimitGfx: $TDC_MAX
  FreqTableGfx:
    FreqTableGfx 8: $GPU_MAX
  FreqTableUclk:
    FreqTableUclk 2: $MEM_MAX
    FreqTableUclk 3: $MEM_MAX
  DcModeMaxFreq:
    DcModeMaxFreq 0: $GPU_MAX
' > mi50-oc.yaml.tpl

# Stock values for 016.004.000.064.016969
echo '
export MEM_MAX=1000
export GPU_MAX=1725
export TDP_MAX=225
export TDC_MAX=330
' > preset-stock.sh
chmod +x preset-stock.sh

# OC worked on all my 4 cards
echo '
export MEM_MAX=1200
export GPU_MAX=1900
export TDP_MAX=300
export TDC_MAX=330
' > preset-oc.sh
chmod +x preset-oc.sh

# Apply preset script
echo '
set -e
PRESET="./preset-$1.sh"
AMDGPU_DEVICE="$2"
. "$PRESET"
if [ ! -e "/sys/bus/pci/devices/$AMDGPU_DEVICE" ]; then
    AMDGPU_DEVICE="0000:$AMDGPU_DEVICE"
fi
FILE=mi50-oc.yaml
envsubst < $FILE.tpl > $FILE
upp -p "/sys/bus/pci/devices/$AMDGPU_DEVICE/pp_table" undump -d $FILE -w
' > apply-preset.sh
chmod +x apply-preset.sh

# Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Vega 20 [Radeon Pro VII/Radeon Instinct MI50 32GB] (rev 01)
AMDGPU_DEVICES=(
  '19:00.0'
  '1c:00.0'
  '33:00.0'
  '4d:00.0'
)
for (( i=0; i<${#AMDGPU_DEVICES[@]}; i++ )); do
  sudo ./apply-preset.sh oc "${AMDGPU_DEVICES[$i]}"
done


How to check errs?

$ sudo rocm-smi --showrasinfo
========================= ROCm System Management Interface =========================
===================================== RAS Info =====================================

GPU[0]:         RAS INFO
         Block       Status    Correctable Error  Uncorrectable Error  
           UMC        ENABLED                  0                    0  
          SDMA        ENABLED                  0                    0  
           GFX        ENABLED                  0                    0  
         MMHUB        ENABLED                  0                    0  
         ATHUB        ENABLED  
      PCIE_BIF        ENABLED                  0                    0  
           HDP        ENABLED                  0                    0  
     XGMI_WAFL       DISABLED  
            DF        ENABLED  
           SMN        ENABLED  
           SEM        ENABLED  
           MP0        ENABLED  
           MP1        ENABLED  
          FUSE        ENABLED  
____________________________________________________________________________________

GPU[1]:         RAS INFO
...

Changing smcPPTable/TdcLimitGfx 350 => 150 reduced the hotspot by 10+- degrees with almost no drop in performance in vllm

temperatures

Results
MEM_MAX=1000; GPU_MAX=1725; TDP_MAX=225; TDC_MAX=330
model	size	params	backend	ngl	n_ubatch	sm	fa	test	t/s
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048	430.10 ± 0.09
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256	32.43 ± 0.02
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048 @ d16384	358.33 ± 13.54
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256 @ d16384	29.74 ± 1.54
MEM_MAX=1150; GPU_MAX=1850; TDP_MAX=300; TDC_MAX=330
model	size	params	backend	ngl	n_ubatch	sm	fa	test	t/s
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048	457.45 ± 0.12
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256	34.05 ± 2.00
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048 @ d16384	389.94 ± 1.68
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256 @ d16384	30.82 ± 1.05
MEM_MAX=1150; GPU_MAX=1850; TDP_MAX=180; TDC_MAX=330
model	size	params	backend	ngl	n_ubatch	sm	fa	test	t/s
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048	441.30 ± 1.18
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256	33.85 ± 0.14
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048 @ d16384	372.79 ± 2.24
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256 @ d16384	31.97 ± 0.21
MEM_MAX=1150; GPU_MAX=1850; TDP_MAX=140; TDC_MAX=330
model	size	params	backend	ngl	n_ubatch	sm	fa	test	t/s
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048	415.37 ± 0.51
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256	32.40 ± 0.13
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048 @ d16384	350.42 ± 1.33
gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256 @ d16384	30.71 ± 0.18
Edit this page
Last updated on May 13, 2026 by mixa3607
Previous
ROCm * graphs * GPUs bench

## K-quant dense-fusion flag (mx-org-densefuse, uncommitted, 2026-09-26)

Env-Flag `GGML_CUDA_REPACK_KQUANT_DENSE_FUSION=1`, gefunden in `/opt/mx-org-densefuse` auf
192.168.178.71, Branch `dense-kquant-fusion`, Basis Org-Commit `5542318e7`. Nicht committed
(reiner Working-Tree-Patch), nicht in mxorig/master enthalten (`git log -S` liefert nichts).

Patch: 13 Zeilen in `ggml/src/ggml-cuda/q8_repack/repack-common.cu`. Erweitert
`ggml_cuda_repack_mmv_fusion_supported` (bisher nur Q8_0/MXFP4) um Q4_K/Q5_K/Q6_K/IQ4_NL,
gesteuert per Env-Var-Gate. Verschmilzt Repack- und Mat-Vec-Kernel zu einem Aufruf statt zwei
(spart Kernel-Launch + Zwischenspeicher-Roundtrip) — wirkt nur beim Single-Token-Decode
(Mat-Vec), nicht beim Prompt-Processing (Mat-Mat, Batch).

Messung: Qwen-35B-A3B MoE, Q5_K_M, 2x MI50 (16G+32G), ROCm, ngl=999:

| Modus | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| Flag OFF (Baseline) | 962.69 ± 12.98 | 63.60 ± 0.12 |
| Flag ON | 963.49 ± 13.67 | 65.70 ± 0.16 |

→ pp512 unverändert (im Rauschen), tg128 **+3.3%**, außerhalb der Standardabweichung.
Kostenloser Decode-Speedup per Flag, aber experimentell/unvalidiert — daher nicht default-on.

## exabit-io/mx-llama.cpp: gfx906 max-ilp scheduler (2026-09-28)

Geklont nach `/opt/exabit-mx-llama.cpp` auf 192.168.178.71 (`exabit-io/mx-llama.cpp`,
`master` @ `82868aa3b`). Gebaut mit `mx-compile.sh` + `-DGGML_HIP_GFX906_MAX_ILP=OFF`
noetig fuer ROCm 6.3.4 — der Default `ON` fuegt `-mllvm -amdgpu-sched-strategy=max-ilp`
hinzu, was der hier verfuegbare Clang nicht kennt (das exabit-Setup faehrt ROCm 10.0).
RCCL ist im Build per `mx-compile.sh` bereits aktiviert (`GGML_HIP_RCCL=ON`).

Vergleich Gemma4-26B-A4B Q6_K_XL, f16 KV, zwei MI50 (32GB+16GB):

| Split | Build | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| 2/1 | upstream (plain) | 928.69 | 59.87 |
| 2/1 | mx-org (eefc4e732) | 1151.22 ± 16.20 | 68.91 |
| 2/1 | exabit | 1089.70 ± 141.59 | 71.38 |
| 3/1 | upstream (plain) | 934.33 | 61.37 |
| 3/1 | mx-org (eefc4e732) | 1146.94 ± 11.18 | 70.24 |
| 3/1 | exabit | 1153.97 ± 13.21 | 71.43 |

pp512: mx-org und exabit praktisch gleichauf (~1150 t/s), beide +24-27% vor upstream.
tg128: exabit gewinnt klar — +2.2 bis +3.6% vor mx-org, +16-19% vor upstream.
Bestes Gesamt-Setup: exabit-Build, tensor-split 3/1. `server`-Macro in
`/opt/llama.cpp/config.yaml` zeigt seit 2026-09-28 auf
`/opt/exabit-mx-llama.cpp/build/bin/llama-server`.

**Korrektur (2026-09-28, spaeter):** `-DGGML_HIP_GFX906_MAX_ILP=OFF` war fuer
den erfolgreichen exabit-Build *noetig* (siehe unten) — der max-ilp-Scheduler
war in den obigen Zahlen also gar nicht aktiv. Der gemessene Gewinn kommt aus
den anderen 17 exabit-Patches (MMVQ-Q8-Fastpath, Norm+Add-Fusion,
GDN-Producer-Fold, DPP-Warp-Reductions, MMVQ-16-Column), nicht aus max-ilp.

## GGML_HIP_GFX906_MAX_ILP: auf diesem ROCm-6.3.4-Toolchain nicht baubar

Portierversuch auf plain Upstream (`/opt/llama.cpp`, reiner CMake-Patch, siehe
`git show 82868aa3b -- ggml/CMakeLists.txt ggml/src/ggml-hip/CMakeLists.txt`)
schlaegt mit exakt demselben Fehler fehl wie zuvor beim exabit-Build:

```
clang (LLVM option parsing): Unknown command line argument '-amdgpu-sched-strategy=max-ilp'.
```

Der hier installierte Clang (ROCm 6.3.4) kennt diese LLVM-Option schlicht
nicht — das exabit-Setup faehrt laut deren README ROCm 10.0. Patch wieder
zurueckgenommen (`git checkout -- ggml/CMakeLists.txt
ggml/src/ggml-hip/CMakeLists.txt`). Ohne ROCm-Upgrade auf diesem Host nicht
nutzbar, weder bei exabit noch bei plain Upstream.

## HSA_XNACK=0 ist Pflicht, steht aber in keiner Shell-Init-Datei (2026-09-28)

`/opt/llama.cpp/mx-compile.sh` (eigenes, ausgereifteres Build-Skript neben
dem mx-org-`mx-compile.sh`, baut plain Upstream bereits mit
`GGML_HIP_RCCL=ON` als Default) dokumentiert:

```
export HSA_XNACK=0        # PFLICHT: sonst melden sich die Karten als
                           # xnack+, rocBLAS hat dafuer keinen Sgemm-Kernel,
                           # MoE-Modelle stuerzen beim Prefill ab
export HSA_P2P_DISABLE=1  # X99: kein PLX-Switch zwischen den MI50-Root-Ports
export HIP_FORCE_P2P_DISABLE=1
```

Der laufende `llama-swap`-Prozess (PID per `pgrep -af llama-swap`) hat diese
Variablen tatsaechlich gesetzt (`cat /proc/<pid>/environ`), aber **in keiner
.bashrc/.profile/systemd-Unit** — vermutlich manuell in einer offen gehaltenen
Shell exportiert, bevor `llama-swap` gestartet wurde. Der Kommentar in
`config.yaml` ("siehe .bashrc: HSA_P2P_DISABLE=1") ist veraltet/falsch, es
gibt keine solche Zeile in `~/.bashrc`.

Alle `llama-bench`-Ad-hoc-Messungen dieser Session (2026-09-26..28) liefen
**ohne** `HSA_XNACK=0` — nur mit `HSA_OVERRIDE_GFX_VERSION`,
`HIP_VISIBLE_DEVICES`, `GGML_HIP_GRAPHS`, `GGML_HIP_ALLOC_GRAPH` gesetzt.
Nichts ist gecrasht, aber die Zahlen sind insofern nicht 1:1 mit einer
Produktionsumgebung vergleichbar, die das Flag konsequent setzt. Fuer
reproduzierbare Zukunftsmessungen: `HSA_XNACK=0` immer mitsetzen, da
nicht-interaktives SSH `.bashrc` ohnehin nicht liest.

Empfehlung (noch nicht umgesetzt): `llama-swap` ueber eine systemd-Unit mit
festen `Environment=`-Zeilen statt einer manuell offen gehaltenen Shell
starten, damit der Prozess einen Neustart des Hosts uebersteht ohne dass die
Pflicht-Variablen verloren gehen.