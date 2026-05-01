#!/usr/bin/env bash
#
# terraform_bootstrap.sh — one-shot Phase-12 bootstrap (PLAN.md §622).
#
# Usage:
#   export YC_TOKEN=<IAM token from `yc iam create-token`>     # or YC_SERVICE_ACCOUNT_KEY_FILE=/path/key.json
#   export TF_VAR_cloud_id=<yc config get cloud-id>
#   export TF_VAR_folder_id=<yc config get folder-id>
#   # Optional (Phase 12 DataSphere bootstrap):
#   #   export TF_VAR_enable_datasphere=true
#   #   export TF_VAR_organization_id=<yc organization-manager organization list>
#   #   export TF_VAR_billing_account_id=<yc billing account list>
#   bash scripts/terraform_bootstrap.sh apply
#
# Modes (positional arg, default = `plan`):
#   plan      — show what would change, never touches Yandex Cloud
#   apply     — provision + emit outputs.json
#   destroy   — tear down (use carefully — drops the SA, registry, bucket!)
#
# Authentication:
#   Either YC_TOKEN (IAM token, 12h lifetime) **or**
#   YC_SERVICE_ACCOUNT_KEY_FILE (service-account JSON key, recommended for CI)
#   must be set. The Yandex Terraform provider reads them automatically.

set -euo pipefail

ACTION="${1:-plan}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${HERE}/infra/terraform"

require() {
  local var="$1"
  if [[ -z "${!var:-}" ]]; then
    echo "[bootstrap] missing required env var: ${var}" >&2
    exit 1
  fi
}

require TF_VAR_cloud_id
require TF_VAR_folder_id

if [[ -z "${YC_TOKEN:-}" && -z "${YC_SERVICE_ACCOUNT_KEY_FILE:-}" ]]; then
  echo "[bootstrap] either YC_TOKEN or YC_SERVICE_ACCOUNT_KEY_FILE must be set" >&2
  exit 1
fi

cd "${TF_DIR}"

if [[ ! -d .terraform ]]; then
  echo "[bootstrap] running terraform init..."
  terraform init -input=false
fi

case "${ACTION}" in
  plan)
    terraform plan -input=false -out=tfplan
    ;;
  apply)
    terraform apply -input=false -auto-approve
    terraform output -json > "${HERE}/runs/terraform_outputs.json"
    echo "[bootstrap] outputs written to runs/terraform_outputs.json"
    ;;
  destroy)
    echo "[bootstrap] DESTROY confirmed in 5s — Ctrl-C to abort..."
    sleep 5
    terraform destroy -input=false -auto-approve
    ;;
  *)
    echo "[bootstrap] unknown action: ${ACTION} (expected plan|apply|destroy)" >&2
    exit 2
    ;;
esac
