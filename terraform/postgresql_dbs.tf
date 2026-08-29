resource "random_password" "f1_predictor_role" {
  length  = 20
  special = false
}

resource "postgresql_role" "f1_predictor" {
  name     = "f1_predictor"
  login    = true
  password = random_password.f1_predictor_role.result
}

resource "postgresql_database" "f1_predictor" {
  name  = "f1_predictor"
  owner = postgresql_role.f1_predictor.name
}

output "f1_predictor_role_password" {
  value     = random_password.f1_predictor_role.result
  sensitive = true
}
