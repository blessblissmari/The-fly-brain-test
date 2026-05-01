# Terraform — FlyBrain on Yandex Cloud

Phase 0 ships the skeleton. Real `apply` lands in Phase 12 (or earlier, before
Phase 7 if you need real LLM traces).

## Prerequisites

* Terraform `>= 1.5`
* Yandex Cloud CLI (`yc`) authenticated
* Cloud + folder IDs available as environment variables:

```bash
export YC_CLOUD_ID=$(yc config get cloud-id)
export YC_FOLDER_ID=$(yc config get folder-id)
```

## Layout

* `versions.tf` — provider pinning.
* `variables.tf` — `cloud_id`, `folder_id`, `zone`, optional name overrides.
* `main.tf` — service account + roles + Container Registry + S3 bucket.
* `outputs.tf` — IDs / bucket / S3 access keys (sensitive).

## Usage

```bash
cd infra/terraform
terraform init
terraform plan \
  -var "cloud_id=$YC_CLOUD_ID" \
  -var "folder_id=$YC_FOLDER_ID"

# When you are ready (Phase 12 or before Phase 7):
terraform apply \
  -var "cloud_id=$YC_CLOUD_ID" \
  -var "folder_id=$YC_FOLDER_ID"
```

## Bootstrap helper

For a one-shot apply with output capture, use `scripts/terraform_bootstrap.sh`:

```bash
export YC_TOKEN="$(yc iam create-token)"   # or YC_SERVICE_ACCOUNT_KEY_FILE=/path/key.json
export TF_VAR_cloud_id=$(yc config get cloud-id)
export TF_VAR_folder_id=$(yc config get folder-id)

bash scripts/terraform_bootstrap.sh plan
bash scripts/terraform_bootstrap.sh apply    # writes runs/terraform_outputs.json
```

## DataSphere (Phase 12)

DataSphere community + project provisioning is opt-in (off by default so a
plain `apply` only creates the minimum-viable stack: SA + roles + Container
Registry + S3 bucket). To enable:

```bash
export TF_VAR_enable_datasphere=true
export TF_VAR_organization_id=$(yc organization-manager organization list --format json | jq -r '.[].id')
export TF_VAR_billing_account_id=$(yc billing account list           --format json | jq -r '.[].id')

bash scripts/terraform_bootstrap.sh apply
```

The corresponding resources (`yandex_datasphere_community`,
`yandex_datasphere_project`) live in `datasphere.tf`. Their IDs are
emitted as outputs `datasphere_community_id` and `datasphere_project_id`.

A starter copy of `terraform.tfvars` is provided as
`terraform.tfvars.example` (gitignored once renamed).

## Note

Phase 0 does NOT run `terraform apply`. The repository ships only the
declarative spec; the user opts in explicitly before incurring any cloud
spend. The Phase-12 stretch is to actually `apply` against a live
Yandex Cloud account once `cloud_id` + an IAM token (or service-account
key) are available.
