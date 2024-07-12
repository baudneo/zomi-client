#!/usr/bin/env bash
trap 'cleanup' SIGINT SIGTERM

cleanup() {
  exit 1
}

if [[ -z "$1" ]]; then
  echo "An event ID MUST be passed as ARG 1, Monitor ID for ARG 2 is optional"
  exit 1
  else
    EID="--eid $1"
fi
if [[ -n "$2" ]]; then
  MID="--mid $2"
  else
    MID=""
fi
echo "ML_CLIENT_CONF_FILE: ${ML_CLIENT_CONF_FILE}"
config="${ML_CLIENT_CONF_FILE:-/opt/zomi/client/conf/client.yml}"
detect_script="${ML_CLIENT_EVENT_START:-$(which zomi-eventproc)}"

event_start_command=(
#  python3
  "${detect_script}"
  --config "${config}"
  --event-mode
  --live
  --event-start
  "${EID}"
  "${MID}"
)

echo "Running command: ${event_start_command[*]}"

eval "${event_start_command[@]}"
exit 0
