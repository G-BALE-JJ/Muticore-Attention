#!/usr/bin/env bash

attention_ui_supports_fancy() {
  if [[ "${GOLEM_ATTENTION_FORCE_TTY:-0}" == 1 ]]; then
    return 0
  fi
  [[ -t 1 && "${TERM:-dumb}" != dumb ]]
}

attention_ui_supports_color() {
  if [[ "${GOLEM_ATTENTION_COLOR:-}" == 1 ]]; then
    return 0
  fi
  if [[ "${GOLEM_ATTENTION_COLOR:-}" == 0 ]]; then
    return 1
  fi
  attention_ui_supports_fancy &&
    [[ "${TERM:-dumb}" != dumb ]] && [[ -z "${NO_COLOR:-}" ]]
}

attention_ui_color() {
  local code="$1"
  shift
  if attention_ui_supports_color; then
    printf '\033[%sm%s\033[0m' "$code" "$*"
  else
    printf '%s' "$*"
  fi
}

attention_ui_format_elapsed() {
  local total="${1:-0}"
  printf '%02d:%02d:%02d' \
    "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

attention_ui_terminal_columns() {
  local columns="${GOLEM_ATTENTION_COLUMNS:-${COLUMNS:-}}"
  if ! [[ "$columns" =~ ^[1-9][0-9]*$ ]]; then
    columns="$(tput cols 2>/dev/null || true)"
  fi
  if ! [[ "$columns" =~ ^[1-9][0-9]*$ ]]; then
    columns=120
  fi
  printf '%s' "$columns"
}

attention_ui_render_progress() {
  local progress="$1"
  local component="$2"
  local phase="$3"
  local elapsed="$4"
  local spinner_index="$5"
  local columns width detail_width layout
  columns="$(attention_ui_terminal_columns)"
  if (( columns >= 110 )); then
    layout=wide
    width=20
    detail_width=$((columns - 61))
  elif (( columns >= 60 )); then
    layout=compact
    width=8
    detail_width=$((columns - 31))
  elif (( columns >= 30 )); then
    layout=minimal
    width=0
    detail_width=$((columns - 20))
  else
    layout=tiny
    width=0
    detail_width=0
  fi
  local filled=$((progress * width / 100))
  local empty=$((width - filled))
  local filled_segment empty_segment bar spinner phase_display percent_display
  filled_segment="$(printf '%*s' "$filled" '' | tr ' ' '=')"
  empty_segment="$(printf '%*s' "$empty" '' | tr ' ' '.')"
  if (( width == 0 )); then
    bar=""
  elif attention_ui_supports_color; then
    printf -v bar '[\033[38;5;48m%s\033[2;37m%s\033[0m]' \
      "$filled_segment" "$empty_segment"
  else
    printf -v bar '[%s%s]' "$filled_segment" "$empty_segment"
  fi
  spinner="${ATTENTION_UI_SPINNERS:="|/-\\"}"
  spinner="${spinner:$((spinner_index % 4)):1}"
  local component_color='1;36'
  case "$component" in
    DISPATCH) component_color='1;34' ;;
    QK) component_color='1;35' ;;
    SOFTMAX) component_color='1;33' ;;
    PV) component_color='1;32' ;;
    DMA) component_color='1;34' ;;
    FAIL|ERROR) component_color='1;31' ;;
  esac
  if [[ "$layout" != wide ]]; then
    phase="${phase// | / }"
  fi
  if [[ "$layout" == minimal ]]; then
    phase="${phase//Softmax/SMX}"
  fi
  if (( detail_width > 0 )); then
    printf -v phase_display "%-${detail_width}.${detail_width}s" "$phase"
  else
    phase_display=""
  fi
  printf -v percent_display '%3d%%' "$progress"
  local component_display
  printf -v component_display '%-9.9s' "$component"
  case "$layout" in
    wide)
      printf '\r  %s %s %s %s  %s  %s %s\033[K' \
        "$(attention_ui_color "$component_color" "$component_display")" \
        "$(attention_ui_color '1;33' "$spinner")" "$bar" \
        "$(attention_ui_color '1;32' "$percent_display")" \
        "$(attention_ui_color '0;37' "$phase_display")" \
        "$(attention_ui_color '2;37' 'elapsed')" \
        "$(attention_ui_color '1;33' "$(attention_ui_format_elapsed "$elapsed")")"
      ;;
    compact)
      printf '\r  %s %s %s %s  %s\033[K' \
        "$(attention_ui_color "$component_color" "$component_display")" \
        "$(attention_ui_color '1;33' "$spinner")" "$bar" \
        "$(attention_ui_color '1;32' "$percent_display")" \
        "$(attention_ui_color '0;37' "$phase_display")"
      ;;
    minimal)
      printf '\r  %s %s %s  %s\033[K' \
        "$(attention_ui_color "$component_color" "$component_display")" \
        "$(attention_ui_color '1;33' "$spinner")" \
        "$(attention_ui_color '1;32' "$percent_display")" \
        "$(attention_ui_color '0;37' "$phase_display")"
      ;;
    tiny)
      local tiny_component_width=$((columns - 5))
      if (( tiny_component_width < 1 )); then tiny_component_width=1; fi
      printf -v component_display "%-${tiny_component_width}.${tiny_component_width}s" "$component"
      printf '\r%s %s\033[K' \
        "$(attention_ui_color "$component_color" "$component_display")" \
        "$(attention_ui_color '1;32' "$percent_display")"
      ;;
  esac
}

attention_ui_stage_result() {
  local label="$1"
  local status="$2"
  local elapsed="$3"
  local status_color=32
  [[ "$status" == PASS ]] || status_color=31
  printf '%s %-28s %s %s\n' \
    "$(attention_ui_color "1;$status_color" "[$status]")" \
    "$(attention_ui_color '1;36' "$label")" \
    "$(attention_ui_color '2;37' 'wall')" \
    "$(attention_ui_color '1;33' "$elapsed")"
}

attention_ui_key_value() {
  local label="$1"
  local value="$2"
  printf '%s %s\n' \
    "$(attention_ui_color '1;36' "$label:")" \
    "$(attention_ui_color '1;34' "$value")"
}

attention_ui_forward_signal() {
  local signal_name="$1"
  local signal_status="$2"
  local command_pid="$3"
  ATTENTION_UI_SIGNAL_STATUS="$signal_status"
  if [[ -n "$command_pid" ]]; then
    kill -s "$signal_name" -- "-$command_pid" 2>/dev/null ||
      kill -s "$signal_name" "$command_pid" 2>/dev/null || true
  fi
}

attention_ui_restore_trap() {
  local signal_name="$1"
  local saved_trap="$2"
  trap - "$signal_name"
  if [[ -n "$saved_trap" ]]; then
    eval "$saved_trap"
  fi
}

attention_ui_run_sst() {
  local stage_log="$1"
  local progress_filter="$2"
  shift 2

  if ! attention_ui_supports_fancy; then
    "$@" 2>&1 | tee "$stage_log" | awk -f "$progress_filter"
    return $?
  fi

  local command_pid=""
  local ATTENTION_UI_SIGNAL_STATUS=0
  local saved_int saved_term saved_hup saved_exit
  saved_int="$(trap -p INT)"
  saved_term="$(trap -p TERM)"
  saved_hup="$(trap -p HUP)"
  saved_exit="$(trap -p EXIT)"
  # Cover the setup window before the signal-specific handlers are installed.
  trap 'ATTENTION_UI_SIGNAL_STATUS=143' INT TERM HUP EXIT
  trap 'attention_ui_forward_signal INT 130 "$command_pid"' INT
  trap 'attention_ui_forward_signal TERM 143 "$command_pid"' TERM
  trap 'attention_ui_forward_signal HUP 129 "$command_pid"' HUP
  trap 'attention_ui_forward_signal TERM 143 "$command_pid"' EXIT

  setsid "$@" > "$stage_log" 2>&1 &
  command_pid=$!

  local start_seconds=$SECONDS
  local progress=0
  local component="SETUP"
  local phase="Starting SST"
  local spinner_index=0
  local refresh_seconds="${GOLEM_ATTENTION_TERMINAL_REFRESH_SECONDS:-0.1}"
  local latest
  while (( ATTENTION_UI_SIGNAL_STATUS == 0 )) &&
    kill -0 "$command_pid" 2>/dev/null; do
    latest=""
    if [[ -n "${ATTENTION_UI_MILESTONE_FILTER:-}" &&
          -f "${ATTENTION_UI_PROGRESS_LOG:-}" ]]; then
      latest="$(awk -v ui=1 -f "$ATTENTION_UI_MILESTONE_FILTER" \
        "$ATTENTION_UI_PROGRESS_LOG" 2>/dev/null | tail -n 1)"
    fi
    if [[ "$latest" =~ ^([0-9]+)\|([^|]+)\|(.+)$ ]]; then
      progress="${BASH_REMATCH[1]}"
      component="${BASH_REMATCH[2]}"
      phase="${BASH_REMATCH[3]}"
    else
      latest="$(awk -f "$progress_filter" "$stage_log" 2>/dev/null | tail -n 1)"
      if [[ "$latest" =~ SST[[:space:]]+([0-9]+)%[[:space:]]+(.+) ]]; then
        progress="${BASH_REMATCH[1]}"
        if (( progress <= 2 )); then component="SETUP"; else component="SST"; fi
        phase="${BASH_REMATCH[2]}"
      fi
    fi
    attention_ui_render_progress \
      "$progress" "$component" "$phase" \
      "$((SECONDS - start_seconds))" "$spinner_index"
    spinner_index=$((spinner_index + 1))
    sleep "$refresh_seconds"
  done

  local status=0
  if (( ATTENTION_UI_SIGNAL_STATUS != 0 )); then
    local shutdown_polls=0
    while kill -0 "$command_pid" 2>/dev/null && (( shutdown_polls < 50 )); do
      sleep 0.1
      shutdown_polls=$((shutdown_polls + 1))
    done
    if kill -0 "$command_pid" 2>/dev/null; then
      kill -KILL -- "-$command_pid" 2>/dev/null ||
        kill -KILL "$command_pid" 2>/dev/null || true
    fi
  fi
  wait "$command_pid" || status=$?
  if (( ATTENTION_UI_SIGNAL_STATUS != 0 )); then
    status="$ATTENTION_UI_SIGNAL_STATUS"
  fi

  attention_ui_restore_trap INT "$saved_int"
  attention_ui_restore_trap TERM "$saved_term"
  attention_ui_restore_trap HUP "$saved_hup"
  attention_ui_restore_trap EXIT "$saved_exit"

  if (( status == 0 )); then
    attention_ui_render_progress \
      100 "COMPLETE" "Attention simulation complete" \
      "$((SECONDS - start_seconds))" 0
  else
    attention_ui_render_progress \
      "$progress" "ERROR" "Attention simulation failed" \
      "$((SECONDS - start_seconds))" 0
  fi
  printf '\n'
  return "$status"
}
