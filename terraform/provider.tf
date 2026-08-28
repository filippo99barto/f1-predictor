terraform {
  required_version = ">= 1.15"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    postgresql = {
      source  = "cyrilgdn/postgresql"
      version = "~> 1.22"
    }
  }
}

provider "aws" {
  region  = "eu-west-2"
  profile = "f1-predictor"

  # Hard stop if credentials ever resolve to the wrong account —
  # paste the Account ID from running:
  # $ aws sts get-caller-identity --profile f1-predictor
  allowed_account_ids = [var.aws_account_id]

  default_tags {
    tags = {
      Project     = "f1-predictor"
      ManagedBy   = "terraform"
    }
  }
}

provider "postgresql" {
  host             = aws_db_instance.postgres.address
  port             = aws_db_instance.postgres.port
  username         = aws_db_instance.postgres.username
  password         = random_password.rds_master.result
  sslmode          = "require"
  superuser        = false
  connect_timeout  = 15
}