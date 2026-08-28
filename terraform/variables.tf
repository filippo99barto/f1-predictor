# $ aws sts get-caller-identity --profile f1-predictor
variable "aws_account_id" {
  description = "AWS account ID Terraform is allowed to operate against"
  type        = string
}