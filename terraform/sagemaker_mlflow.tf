resource "aws_iam_role" "mlflow_app" {
  name = "f1-predictor-mlflow-app"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "mlflow_app_s3" {
  name = "mlflow-app-s3-access"
  role = aws_iam_role.mlflow_app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.artifacts.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "mlflow_app_model_registry" {
  name = "mlflow-app-model-registry"
  role = aws_iam_role.mlflow_app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:ListModelPackageGroups",
          "sagemaker:CreateModelPackage",
          "sagemaker:DescribeModelPackage",
          "sagemaker:UpdateModelPackage",
          "sagemaker:ListModelPackages",
          "sagemaker:AddTags",
          "sagemaker:ListTags"
        ]
        Resource = [
          "arn:aws:sagemaker:eu-west-2:105257729461:model-package-group/f1-*",
          "arn:aws:sagemaker:eu-west-2:105257729461:model-package/f1-*/*"
        ]
      }
    ]
  })
}

resource "aws_sagemaker_mlflow_app" "mlflow" {
  name                     = "f1-predictor-mlflow"
  artifact_store_uri       = "s3://${aws_s3_bucket.artifacts.id}/mlflow"
  role_arn                 = aws_iam_role.mlflow_app.arn
  model_registration_mode  = "AutoModelRegistrationEnabled"
}

output "mlflow_app_arn" {
  value = aws_sagemaker_mlflow_app.mlflow.arn
}