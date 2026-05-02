variable "cloud_id" {
  type        = string
  description = "Yandex Cloud cloud_id."
}

variable "folder_id" {
  type        = string
  description = "Yandex Cloud folder_id where FlyBrain resources live."
}

variable "zone" {
  type        = string
  default     = "ru-central1-a"
  description = "Yandex Cloud availability zone."
}

variable "service_account_name" {
  type    = string
  default = "flybrain-sa"
}

variable "container_registry_name" {
  type    = string
  default = "flybrain-registry"
}

variable "s3_bucket_name" {
  type        = string
  default     = ""
  description = "Override for the S3 bucket name. Yandex Object Storage bucket names are globally unique across all tenants, so leaving this empty falls back to flybrain-data-<folder_id> which is guaranteed to be free."
}

locals {
  effective_s3_bucket_name = (
    var.s3_bucket_name != "" ? var.s3_bucket_name : "flybrain-data-${var.folder_id}"
  )
}

# DataSphere (opt-in) ---------------------------------------------------------

variable "enable_datasphere" {
  type        = bool
  default     = false
  description = "Set to true to provision a DataSphere community + project (Phase 12)."
}

variable "organization_id" {
  type        = string
  default     = ""
  description = <<-EOT
    Yandex Cloud organization_id; required when enable_datasphere = true.
    Find it with: yc organization-manager organization list --format json
  EOT
}

variable "billing_account_id" {
  type        = string
  default     = ""
  description = <<-EOT
    Yandex Cloud billing account ID; required when enable_datasphere = true.
    Find it with: yc billing account list --format json
  EOT
}

variable "datasphere_community_name" {
  type    = string
  default = "flybrain-optimizer"
}

variable "datasphere_project_name" {
  type    = string
  default = "flybrain-optimizer"
}

variable "datasphere_max_units_per_hour" {
  type    = number
  default = 100
}

variable "datasphere_max_units_per_execution" {
  type    = number
  default = 50
}

variable "datasphere_balance" {
  type    = number
  default = 1000
}
