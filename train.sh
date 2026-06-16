#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
COMMON_CONFIG="${COMMON_CONFIG:-conf/common.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-conf/model.yaml}"
MODE="${MODE:-train}"
CHECKPOINT="${CHECKPOINT:-}"
OVERRIDE_CONFIG="${CONFIG:-}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${MODE}_$(date +%Y%m%d_%H%M%S).log}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python or create .venv first." >&2
  exit 1
fi

set -- \
  "${PYTHON_BIN}" \
  "-m" "core.main" \
  "--common-config" "${COMMON_CONFIG}" \
  "--model-config" "${MODEL_CONFIG}" \
  "--mode" "${MODE}"

if [ -n "${OVERRIDE_CONFIG}" ]; then
  set -- "$@" "--config" "${OVERRIDE_CONFIG}"
fi

if [ -n "${CHECKPOINT}" ]; then
  set -- "$@" "--checkpoint" "${CHECKPOINT}"
fi

mkdir -p "$(dirname -- "${LOG_FILE}")"

printf "Running:"
printf " %s" "$@"
printf "\n"
printf "Log file: %s\n" "${LOG_FILE}"

"$@" 2>&1 | tee "${LOG_FILE}"
