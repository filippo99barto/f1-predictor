# $ aws sts get-caller-identity --profile f1-predictor
variable "aws_account_id" {
  description = "AWS account ID Terraform is allowed to operate against"
  type        = string
}

# current public IP -> curl -s https://checkip.amazonaws.com
variable "my_ip" {
  description = "Your current public IP in CIDR notation, e.g. 82.14.22.10/32"
  type        = string
}