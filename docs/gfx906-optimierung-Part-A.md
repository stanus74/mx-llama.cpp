# gfx906 MMQ `nwarps`-Optimierung — Anleitung

**Ziel:** Prompt-Processing-Durchsatz auf MI50/MI60 (gfx906) erhöhen, indem die
Warp-Parallelität im quantisierten Matrixmultiplikations-Kernel (MMQ) angehoben wird.

**Ergebnis (gemessen, `/opt/llama.cpp`, Mainline, Commit `e920c523e`):**

| Modell | Quant | pp512 vorher (nwarps=4) | pp512 nachher (nwarps=8) | Δ |
|---|---|---|---|---|
| Qwopus3.6-27B-Coder-Compat-MTP | Q6_K | ~163 t/s | 190.99 ± 0.57 t/s | **+17.3 %** |

Referenzwerte aus einem separaten Sweep (mx-llama.cpp-Fork, `docs/gfx906-optimization-notes.md`):
Q5_K +23 %, Q6_K +19 %, jeweils single-GPU, `nwarps=8` als Sweet Spot (`nwarps=16` regressiert
durch Occupancy-Klippe). Ursprüngliche Beobachtung: llama.cpp-Discussion
[#23881](https://github.com/ggml-org/llama.cpp/discussions/23881) (MI60/MI50, bis zu +76 % bei Q8).

---

## Hintergrund

llama.cpp wählt die Anzahl paralleler Warps pro MMQ-Kernel-Block über eine generische
Heuristik `256 / warp_size`. Diese Heuristik ist für Matrix-Core-Architekturen (MFMA, ab
CDNA/gfx908) kalibriert. **gfx906 (Vega20/MI50/MI60) hat keine Matrix-Cores** — die gesamte
Matrixmultiplikation läuft über normale ALU-Dot-Product-Instruktionen. Dort ist die
Kernel-Konfiguration entscheidender für die Auslastung, und die generische Heuristik
unterdimensioniert die Compute-Units deutlich (`nwarps=4` bei Wave64 statt besser `8`).

## Zentrale Randbedingung

`nwarps` bestimmt Block-Dimensionen und Shared-Memory-Layout **sowohl auf der Host-Seite**
(Kernel-Launch-Parameter) **als auch auf der Device-Seite** (`__launch_bounds__`,
Compile-Time-Konstante). Beide Seiten müssen exakt denselben Wert verwenden — bei einem
Mismatch entstehen ROCm-Launch-Fehler (`unspecified launch failure`), da Block-Dims und
Shared-Memory-Größe nicht mehr zum tatsächlich kompilierten Kernel passen.

> ⚠️ Nur die Host-Funktion zu patchen (ohne die Device-Funktion) führt zu genau diesem Crash
> — real reproduziert während der Entwicklung dieser Änderung.

## Umsetzung

Datei: `ggml/src/ggml-cuda/mmq.cuh`

Ein Macro `GGML_MMQ_NWARPS_GFX906` (Default `8`) steuert beide Stellen synchron:

```cpp
#ifndef GGML_MMQ_NWARPS_GFX906
#define GGML_MMQ_NWARPS_GFX906 8
#endif
```

**Host-Funktion** (`mmq_get_nwarps_host`) — VEGA20-Sonderfall ergänzt:

```cpp
static int mmq_get_nwarps_host(const int cc, const int warp_size) {
    if (amd_mfma_available(cc)) {
        return 8;
    }
    if (cc == GGML_CUDA_CC_VEGA20) {
        return GGML_MMQ_NWARPS_GFX906;
    }
    return 256/warp_size;
}
```

**Device-Funktion** (`mmq_get_nwarps_device`) — `__gfx906__`-Zweig ergänzt:

```cpp
static constexpr __device__ int mmq_get_nwarps_device() {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    return 8;
#elif defined(GGML_USE_HIP) && defined(__gfx906__)
    return GGML_MMQ_NWARPS_GFX906;
#else
    return 256/ggml_cuda_get_physical_warp_size();
#endif
}
```

Alle anderen Architekturen (inkl. CDNA/MFMA-Karten) sind von der Änderung unberührt — der
gfx906-Zweig greift nur bei exakt dieser Architektur.

## Patch anwenden

```bash
cd /opt/llama.cpp   # oder anderes llama.cpp-Root-Verzeichnis

# 1) Macro-Definition
sed -i '/^static int mmq_get_nwarps_host/i\
#ifndef GGML_MMQ_NWARPS_GFX906\
#define GGML_MMQ_NWARPS_GFX906 8\
#endif' ggml/src/ggml-cuda/mmq.cuh

# 2) Host- und Device-Funktion (Python, robuster als sed bei Mehrzeilen-Matches)
python3 - << 'EOF'
path = "ggml/src/ggml-cuda/mmq.cuh"
with open(path) as f:
    content = f.read()

host_old = """static int mmq_get_nwarps_host(const int cc, const int warp_size) {
    return amd_mfma_available(cc) ? 8 : 256/warp_size;
}"""
host_new = """static int mmq_get_nwarps_host(const int cc, const int warp_size) {
    if (amd_mfma_available(cc)) {
        return 8;
    }
    if (cc == GGML_CUDA_CC_VEGA20) {
        return GGML_MMQ_NWARPS_GFX906;
    }
    return 256/warp_size;
}"""

dev_old = """static constexpr __device__ int mmq_get_nwarps_device() {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    return 8;
#else
    return 256/ggml_cuda_get_physical_warp_size();
#endif // AMD_MFMA_AVAILABLE
}"""
dev_new = """static constexpr __device__ int mmq_get_nwarps_device() {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    return 8;
#elif defined(GGML_USE_HIP) && defined(__gfx906__)
    return GGML_MMQ_NWARPS_GFX906;
#else
    return 256/ggml_cuda_get_physical_warp_size();
#endif // AMD_MFMA_AVAILABLE
}"""

assert host_old in content, "Host-Muster nicht gefunden - Datei weicht ab!"
assert dev_old in content, "Device-Muster nicht gefunden - Datei weicht ab!"
content = content.replace(host_old, host_new).replace(dev_old, dev_new)

with open(path, "w") as f:
    f.write(content)
print("Gepatcht.")
EOF

# 3) Gegenpruefen: sollte genau 4 Treffer zeigen (Definition, 2x #ifndef/#define, Host-Nutzung, Device-Nutzung)
grep -n "GGML_MMQ_NWARPS_GFX906" ggml/src/ggml-cuda/mmq.cuh
```

## Bauen

```bash
./mx-compile.sh   # oder aequivalentes Build-Skript mit RCCL=ON, ROCWMMA_FATTN=OFF
```

## Korrektheit verifizieren (Pflicht vor Produktivbetrieb)

```bash
./build/bin/test-backend-ops -o MUL_MAT 2>&1 | grep -i fail
./build/bin/test-backend-ops -o MUL_MAT 2>&1 | grep -i fail
./build/bin/test-backend-ops -o MUL_MAT 2>&1 | grep -i fail
```

Jeder Durchlauf sollte **keine** Ausgabe liefern (kein `FAIL`, kein `ROCm error`). Grund für
die Mehrfachprüfung: in der ursprünglichen Discussion wurde bei `nwarps=8` ein seltenes,
nicht-deterministisches Fehlverhalten bei `q5_1, m=16, n=1, k=32` beobachtet — ein einzelner
grüner Lauf schließt das nicht sicher aus. Laufzeit pro Durchlauf: ca. 1–1,5 Minuten.

## Performance messen

```bash
HIP_VISIBLE_DEVICES=1 ./build/bin/llama-bench \
  -m DEIN_MODELL.gguf \
  -ngl 99 -fa 1 -mmp 0 -dio 1 -r 3 -p 512 -n 128
```

Single-GPU (`HIP_VISIBLE_DEVICES=1`), um den reinen Kernel-Effekt ohne Multi-GPU-Overhead zu
sehen. Vergleichswert vorher/nachher am besten mit demselben Modell vor und nach dem Patch.

## Zurückrollen

Falls Probleme auftreten, Macro auf `4` setzen (deaktiviert die Änderung ohne Code-Revert):

```bash
sed -i 's/#define GGML_MMQ_NWARPS_GFX906 8/#define GGML_MMQ_NWARPS_GFX906 4/' ggml/src/ggml-cuda/mmq.cuh
```

Oder vollständig zurücksetzen:

```bash
git diff ggml/src/ggml-cuda/mmq.cuh   # Aenderung ansehen
git checkout -- ggml/src/ggml-cuda/mmq.cuh
```

## Bekannte Grenzen

- Nur für gfx906 (MI50/MI60/Radeon VII/Radeon Pro VII) — andere Architekturen unberührt
- Getestet mit Q5_K und Q6_K; `nwarps=16` wurde separat getestet und regressiert (Occupancy-
  Klippe auf MI50 32 GB) — nicht pauschal weiter erhöhen ohne erneuten Sweep
- Ein pauschaler Versuch, `nwarps=8` für **alle** Quant-Typen inkl. Q4_0 zu erzwingen, führte
  zu einem reproduzierbaren `ROCm error: unspecified launch failure` bei
  `Q4_0, m=16, n=8, k=256` in einer früheren, fehlerhaften Zwischenversion dieses Patches
  (Host ohne synchronen Device-Patch) — dieser Patch behebt das durch die Synchronisierung,
  wurde aber nur mit Q5_K/Q6_K nachgemessen, nicht mit Q4_0 selbst
- Kein Q8_0-Modell zum Nachmessen verfügbar; laut Discussion #23881 könnte dort sogar
  `nwarps=16` noch besser sein als `8` — offener Punkt für zukünftige Tests