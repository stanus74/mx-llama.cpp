https://arkprojects.space/wiki/AMD_GFX906/pcie-lnk-speed

Perf vs PCIe speed
software: llama.cpp (b9119-rocm-7.2.1)
mobo: imb760
meter: rocm-smi --showbw
lnk ver: pcie_set_speed.sh
More info
Max bw
Link	Workload	BW (mb/s)
x8 1.0	pp	589
x8 1.0	tg	209
x8 2.0	pp	728
x8 2.0	tg	228
x8 3.0	pp	1104
x8 3.0	tg	219
x8 4.0	pp	1135
x8 4.0	tg	222
Bench
MODEL=unsloth/gemma-4-31B-it-GGUF:Q8_0
cpupower -c 0-37 frequency-set -g performance
numactl --membind=0 --cpunodebind=0 \
./llama-bench \
  --hf-repo $MODEL \
  --split-mode tensor --flash-attn 1 \
  --n-prompt 2048 --ubatch-size 2048 \
  --n-gen 256 \
  --n-depth 0,16384

Link	model	size	params	backend	ngl	n_ubatch	sm	fa	test	t/s
x8 1.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048	285.30 ± 0.03
x8 1.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256	28.90 ± 0.02
x8 1.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048 @ d16384	248.42 ± 0.44
x8 1.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256 @ d16384	27.47 ± 0.09
x8 2.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048	360.82 ± 0.08
x8 2.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256	31.39 ± 0.16
x8 2.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048 @ d16384	311.40 ± 0.63
x8 2.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256 @ d16384	29.59 ± 0.11
x8 3.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048	414.26 ± 0.12
x8 3.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256	31.95 ± 0.02
x8 3.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048 @ d16384	355.50 ± 0.75
x8 3.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256 @ d16384	30.27 ± 0.14
x8 4.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048	447.61 ± 0.08
x8 4.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256	32.52 ± 0.06
x8 4.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	pp2048 @ d16384	382.58 ± 0.97
x8 4.0	gemma4 31B Q8_0	30.38 GiB	30.70 B	ROCm	99	2048	tensor	1	tg256 @ d16384	30.67 ± 0.11
Edit this page
Last updated on May 13, 2026 by mixa3607
