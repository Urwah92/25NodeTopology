#!/usr/bin/env bash
# Launch a conf script inside selected containers, detached, with logs.
# Usage examples for distributed sellers:
# One seller: bash start_sellers.sh 2
# Multiple specific sellers: bash start_sellers.sh 2 5 9
# Comma-separated form: bash start_sellers.sh --nodes 2,5,9
# Mixed form also works: bash start_sellers.sh --nodes serf2,serf8,12
# With the default prefix, 2 maps to clab-century-serf2.

set -euo pipefail

PREFIX="clab-century-serf"
START=13
END=15
TARGET_IDS=()
CONTAINER_APP_DIR="/opt/serfapp"
CONF_SCRIPT="./config_ACtop2p.sh"

log() { echo -e "[\e[1mCONF\e[0m] $*"; }
warn() { echo -e "[\e[33mWARN\e[0m] $*" >&2; }
err() { echo -e "[\e[31mERROR\e[0m] $*" >&2; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { err "Required command '$1' not found."; exit 1; }; }
is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }

add_target_id() {
  local raw="$1"
  raw="${raw//[[:space:]]/}"
  local id="${raw#serf}"
  if ! is_integer "$id"; then
    err "Invalid container id '$raw'. Use numbers like 2 or names like serf2."
    exit 1
  fi
  TARGET_IDS+=("$id")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2;;
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --nodes|--containers|--sellers)
      IFS=',' read -r -a raw_ids <<< "$2"
      for raw_id in "${raw_ids[@]}"; do
        [[ -n "$raw_id" ]] || continue
        add_target_id "$raw_id"
      done
      shift 2
      ;;
    --dir|--container-dir) CONTAINER_APP_DIR="$2"; shift 2;;
    --script) CONF_SCRIPT="$2"; shift 2;;
    --help|-h)
      cat <<USAGE
Usage: $0 [--prefix PFX] [--start N] [--end M]
          [--nodes 2,5,9]
          [--dir /opt/serfapp]
          [--script ./config_ACtop2p.sh]
          [2 5 9]

Examples:
  $0 --start 13 --end 15              # contiguous seller range
  $0 --nodes 2,5,9                    # explicit seller ids
  $0 2 5 9                            # positional seller ids
  $0 --nodes serf2,serf8,12           # mixed explicit ids
USAGE
      exit 0
      ;;
    *)
      if [[ "$1" == -* ]]; then
        echo "Unknown flag: $1" >&2
        exit 1
      fi
      add_target_id "$1"
      shift
      ;;
  esac
done

require_cmd docker

container_name() { echo "${PREFIX}$1"; }

file_exists_in_container() {
  local name="$1"; local path="$2"
  docker exec -u root "$name" bash -lc "[ -e '$path' ]"
}

run_bg_in_container_dir() {
  local name="$1"; shift
  local dir="$1"; shift
  local cmd="$*"
  docker exec -u root "$name" bash -lc "mkdir -p /var/log"
  docker exec -u root -d "$name" bash -lc "cd '$dir' && { $cmd >>/var/log/serfapp_bg.log 2>&1 & }"
}

if (( ${#TARGET_IDS[@]} == 0 )); then
  if ! is_integer "$START" || ! is_integer "$END"; then
    err "--start and --end must be integers."
    exit 1
  fi
  if (( START > END )); then
    err "--start must be less than or equal to --end."
    exit 1
  fi

  for i in $(seq "$START" "$END"); do
    TARGET_IDS+=("$i")
  done
fi

for i in "${TARGET_IDS[@]}"; do
  name="$(container_name "$i")"
  if file_exists_in_container "$name" "${CONTAINER_APP_DIR}/${CONF_SCRIPT}"; then
    log "Launching ${CONF_SCRIPT} in $name (detached)..."
    run_bg_in_container_dir "$name" "$CONTAINER_APP_DIR" "bash '${CONF_SCRIPT}'"
  else
    warn "Missing ${CONTAINER_APP_DIR}/${CONF_SCRIPT} in $name; skipping."
  fi
done

log "Done. Tail logs with: docker exec -it $(container_name "${TARGET_IDS[0]}") tail -f /var/log/serfapp_bg.log"
