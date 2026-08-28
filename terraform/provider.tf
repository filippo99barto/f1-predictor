terraform {
  required_version = ">= 1.15"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # No backend block = local state (terraform.tfstate in this folder, gitignored)
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