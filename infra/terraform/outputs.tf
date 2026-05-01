output "service_account_id" {
  value = yandex_iam_service_account.flybrain.id
}

output "container_registry_id" {
  value = yandex_container_registry.flybrain.id
}

output "s3_bucket" {
  value = yandex_storage_bucket.flybrain_data.bucket
}

output "s3_access_key_id" {
  value     = yandex_iam_service_account_static_access_key.flybrain_s3.access_key
  sensitive = true
}

output "datasphere_community_id" {
  value       = var.enable_datasphere ? yandex_datasphere_community.flybrain[0].id : null
  description = "DataSphere community ID (null when enable_datasphere = false)."
}

output "datasphere_project_id" {
  value       = var.enable_datasphere ? yandex_datasphere_project.flybrain[0].id : null
  description = "DataSphere project ID (null when enable_datasphere = false)."
}
