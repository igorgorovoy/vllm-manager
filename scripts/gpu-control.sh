#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Помилка: nvidia-smi не знайдено. Перевір драйвер NVIDIA." >&2
  exit 1
fi

print_help() {
  cat <<'EOF'
Використання:
  gpu-control.sh status
  gpu-control.sh health [--strict] [temp_warn] [temp_crit] [util_warn] [mem_warn] [power_warn] [power_crit]
  gpu-control.sh watch [sec]
  gpu-control.sh dmon
  gpu-control.sh procs
  gpu-control.sh log [interval_sec] [output.csv]
  gpu-control.sh pm on|off
  gpu-control.sh pl <watts>
  gpu-control.sh help

Команди:
  status            Разово показати стан GPU через nvidia-smi.
  health [...]      Швидка оцінка стану: OK/WARN/ALERT.
  watch [sec]       Live-режим (за замовчуванням 1 сек).
  dmon              Потокові метрики (power/util/clocks/mem/temp).
  procs             Показати процеси, що використовують GPU.
  log [i] [file]    Логувати метрики в CSV (i=сек; default 2; file=gpu_log.csv).
  pm on|off         Увімкнути/вимкнути Persistence Mode (зазвичай потрібен sudo).
  pl <watts>        Встановити power limit у ватах (зазвичай потрібен sudo).
  help              Показати цю довідку.
EOF
}

require_sudo_for_control() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Ця команда зазвичай вимагає root. Запусти з sudo." >&2
    exit 1
  fi
}

cmd="${1:-help}"

case "${cmd}" in
  status)
    nvidia-smi
    ;;
  health)
    strict_mode=0
    shift
    if [[ "${1:-}" == "--strict" ]]; then
      strict_mode=1
      shift
    fi

    temp_warn="${1:-75}"
    temp_crit="${2:-85}"
    util_warn="${3:-95}"
    mem_warn="${4:-90}"
    power_warn="${5:-250}"
    power_crit="${6:-320}"

    if ! [[ "${temp_warn}" =~ ^[0-9]+$ && "${temp_crit}" =~ ^[0-9]+$ && "${util_warn}" =~ ^[0-9]+$ && "${mem_warn}" =~ ^[0-9]+$ && "${power_warn}" =~ ^[0-9]+$ && "${power_crit}" =~ ^[0-9]+$ ]]; then
      echo "Параметри health мають бути цілими числами." >&2
      exit 1
    fi

    if (( temp_warn >= temp_crit )); then
      echo "Некоректні пороги: temp_warn має бути менше за temp_crit." >&2
      exit 1
    fi
    if (( power_warn >= power_crit )); then
      echo "Некоректні пороги: power_warn має бути менше за power_crit." >&2
      exit 1
    fi

    mapfile -t rows < <(nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits)
    if (( ${#rows[@]} == 0 )); then
      echo "Не вдалося отримати метрики GPU." >&2
      exit 1
    fi

    overall="OK"
    for row in "${rows[@]}"; do
      IFS=',' read -r gpu_index gpu_name temp util mem_used mem_total power_draw_raw <<<"${row}"
      gpu_index="${gpu_index// /}"
      gpu_name="${gpu_name#"${gpu_name%%[![:space:]]*}"}"
      gpu_name="${gpu_name%"${gpu_name##*[![:space:]]}"}"
      temp="${temp// /}"
      util="${util// /}"
      mem_used="${mem_used// /}"
      mem_total="${mem_total// /}"
      power_draw_raw="${power_draw_raw// /}"

      mem_pct=0
      mem_display="${mem_used}/${mem_total} MiB"
      mem_supported=1
      if ! [[ "${mem_used}" =~ ^[0-9]+$ && "${mem_total}" =~ ^[0-9]+$ ]]; then
        mem_supported=0
        mem_display="N/A"
      elif (( mem_total > 0 )); then
        mem_pct=$(( mem_used * 100 / mem_total ))
      fi

      power_supported=1
      power_watts=0
      power_display="${power_draw_raw}W"
      if [[ "${power_draw_raw}" == "N/A" || "${power_draw_raw}" == "[N/A]" || -z "${power_draw_raw}" ]]; then
        power_supported=0
        power_display="N/A"
      elif [[ "${power_draw_raw}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        power_watts="${power_draw_raw%.*}"
      else
        power_supported=0
        power_display="N/A"
      fi

      state="OK"
      if (( temp >= temp_crit || (power_supported == 1 && power_watts >= power_crit) )); then
        state="ALERT"
      elif (( temp >= temp_warn || util >= util_warn || (power_supported == 1 && power_watts >= power_warn) || (strict_mode == 0 && mem_supported == 1 && mem_pct >= mem_warn) )); then
        state="WARN"
      fi

      if [[ "${state}" == "ALERT" ]]; then
        overall="ALERT"
      elif [[ "${state}" == "WARN" && "${overall}" == "OK" ]]; then
        overall="WARN"
      fi

      status_line="GPU ${gpu_index} (${gpu_name}): ${state} | temp=${temp}C util=${util}% power=${power_display}"
      if (( strict_mode == 0 )); then
        if (( mem_supported == 1 )); then
          status_line="${status_line} mem=${mem_display} (${mem_pct}%)"
        else
          status_line="${status_line} mem=N/A"
        fi
      fi
      echo "${status_line}"
    done

    echo "Загальний стан: ${overall}"
    if (( strict_mode == 1 )); then
      echo "Режим: strict (ігнорує VRAM)"
      echo "Пороги: temp_warn=${temp_warn}C temp_crit=${temp_crit}C util_warn=${util_warn}% power_warn=${power_warn}W power_crit=${power_crit}W"
    else
      echo "Режим: standard"
      echo "Пороги: temp_warn=${temp_warn}C temp_crit=${temp_crit}C util_warn=${util_warn}% mem_warn=${mem_warn}% power_warn=${power_warn}W power_crit=${power_crit}W"
    fi
    ;;
  watch)
    interval="${2:-1}"
    watch -n "${interval}" nvidia-smi
    ;;
  dmon)
    nvidia-smi dmon -s pucmt
    ;;
  procs)
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits || true
    echo
    echo "Повний список графічних/compute процесів дивись у звичайному nvidia-smi:"
    nvidia-smi
    ;;
  log)
    interval="${2:-2}"
    out="${3:-gpu_log.csv}"
    echo "Логування в ${out} (інтервал ${interval}s). Зупинка: Ctrl+C"
    nvidia-smi \
      --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
      --format=csv \
      -l "${interval}" > "${out}"
    ;;
  pm)
    mode="${2:-}"
    case "${mode}" in
      on)
        require_sudo_for_control
        nvidia-smi -pm 1
        ;;
      off)
        require_sudo_for_control
        nvidia-smi -pm 0
        ;;
      *)
        echo "Невірний параметр для pm. Використай: on або off" >&2
        exit 1
        ;;
    esac
    ;;
  pl)
    watts="${2:-}"
    if [[ -z "${watts}" ]]; then
      echo "Вкажи значення у ватах. Приклад: gpu-control.sh pl 220" >&2
      exit 1
    fi
    require_sudo_for_control
    nvidia-smi -pl "${watts}"
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    echo "Невідома команда: ${cmd}" >&2
    print_help
    exit 1
    ;;
esac
