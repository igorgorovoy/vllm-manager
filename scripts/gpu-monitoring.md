# Documentation: NVIDIA GPU monitoring and basic control

This file contains a short set of practical commands for diagnosing GPU state on Linux.

## 1) Quick check

```bash
nvidia-smi
```

What to look at:
- `GPU-Util` — current GPU load (%)
- `Memory-Usage` — memory in use
- `Temp` — temperature
- `Pwr:Usage` — current power draw
- `Processes` — processes using the GPU

## 2) Live monitoring

```bash
watch -n 1 nvidia-smi
```

- Refreshes once per second
- Handy for real-time observation

## 3) Streaming metrics (lightweight)

```bash
nvidia-smi dmon -s pucmt
```

Flags:
- `p` — power
- `u` — utilization
- `c` — clocks
- `m` — memory
- `t` — temperature

Stop with `Ctrl+C`.

## 4) Logging state to CSV

```bash
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 2
```

- `-l 2` — collect metrics every 2 seconds
- Redirect to a file: `... > gpu_log.csv`

## 5) Basic control (needs sudo)

### Enable Persistence Mode

```bash
sudo nvidia-smi -pm 1
```

### Disable Persistence Mode

```bash
sudo nvidia-smi -pm 0
```

### Set power limit (if supported)

```bash
sudo nvidia-smi -pl 250
```

Hint: check the supported range in `nvidia-smi -q -d POWER`.

## 6) Bundled script

In the same directory there is a script `gpu-control.sh` with these commands:
- `status` — one-shot state
- `health [--strict] [temp_warn] [temp_crit] [util_warn] [mem_warn] [power_warn] [power_crit]` — quick `OK/WARN/ALERT` verdict
- `watch [sec]` — live mode
- `dmon` — streaming metrics
- `procs` — list processes
- `log [interval] [file]` — CSV logging
- `pm on|off` — persistence mode
- `pl <watts>` — power limit
- `help` — help text

Examples:

```bash
./gpu-control.sh status
./gpu-control.sh health
./gpu-control.sh health --strict
./gpu-control.sh health 75 85 95 90 250 320
./gpu-control.sh watch 1
./gpu-control.sh log 2 gpu_log.csv
sudo ./gpu-control.sh pm on
sudo ./gpu-control.sh pl 220
```

### The `health` command

Default thresholds:
- `temp_warn=75C`
- `temp_crit=85C`
- `util_warn=95%`
- `mem_warn=90%`
- `power_warn=250W`
- `power_crit=320W`

Logic:
- `ALERT`: temperature `>= temp_crit` or power `>= power_crit`
- `WARN`: temperature `>= temp_warn`, or GPU util `>= util_warn`, or power `>= power_warn`, or VRAM `>= mem_warn` (in the default mode)
- `OK`: none of the above conditions triggered

`--strict` mode:
- Ignores VRAM (useful when `memory.*` returns `N/A`)
- Evaluates state using only `temp + util + power`
