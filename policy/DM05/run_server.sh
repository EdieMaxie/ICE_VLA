#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${POLICY_DIR}/opendm:${POLICY_DIR}/../..${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
exec "${POLICY_DIR}/.venv/bin/python" \
  "${POLICY_DIR}/../../setup_policy_server.py" \
  --config_path "${POLICY_DIR}/deploy.yml"
