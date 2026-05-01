# Optional DataSphere bootstrap (PLAN.md §622 — Phase 12 stretch).
#
# Off by default so a plain `terraform apply` only provisions the
# minimum-viable stack (service account + roles + Container Registry +
# S3 bucket) and never accidentally creates a billable DataSphere
# community / project. Flip ``enable_datasphere = true`` (in
# ``terraform.tfvars``) to opt in.
#
# A DataSphere community is **per-organization**, not per-folder, so
# ``var.organization_id`` and ``var.billing_account_id`` are required
# when ``enable_datasphere`` is on. Look them up with:
#
#   yc organization-manager organization list --format json | jq -r '.[].id'
#   yc billing account list           --format json | jq -r '.[].id'

resource "yandex_datasphere_community" "flybrain" {
  count = var.enable_datasphere ? 1 : 0

  name               = var.datasphere_community_name
  description        = "FlyBrain Optimizer DataSphere community (Phase 12)."
  organization_id    = var.organization_id
  billing_account_id = var.billing_account_id

  labels = {
    project = "flybrain-optimizer"
    phase   = "12"
  }
}

resource "yandex_datasphere_project" "flybrain" {
  count = var.enable_datasphere ? 1 : 0

  name         = var.datasphere_project_name
  description  = "FlyBrain Optimizer DataSphere project (training, eval, traces)."
  community_id = yandex_datasphere_community.flybrain[0].id

  labels = {
    project = "flybrain-optimizer"
    phase   = "12"
  }

  limits = {
    # Conservative defaults — bump if you need to run PPO / full bench.
    max_units_per_hour      = var.datasphere_max_units_per_hour
    max_units_per_execution = var.datasphere_max_units_per_execution
    balance                 = var.datasphere_balance
  }

  settings = {
    service_account_id = yandex_iam_service_account.flybrain.id
    # Default subnet is auto-picked by DataSphere when omitted; leave
    # other settings at their defaults so a fresh apply succeeds with
    # only the required IDs from the user.
  }
}
