#!/bin/sh
set -eu

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
COMMON_CONFIG="${COMMON_CONFIG:-conf/common.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-conf/model.yaml}"
MODE="${MODE:-train}"
CHECKPOINT="${CHECKPOINT:-}"
OVERRIDE_CONFIG="${CONFIG:-}"

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

printf "Running:"
printf " %s" "$@"
printf "\n"
exec "$@"
