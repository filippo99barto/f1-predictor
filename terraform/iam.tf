resource "aws_iam_user" "mlflow_user" {
  name = "f1-predictor-mlflow"
}

resource "aws_iam_access_key" "mlflow_user" {
  user = aws_iam_user.mlflow_user.name
}

resource "aws_iam_user_policy" "mlflow_s3" {
  name = "mlflow-s3-access"
  user = aws_iam_user.mlflow_user.name

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

output "mlflow_access_key_id" {
  value = aws_iam_access_key.mlflow_user.id
}

output "mlflow_secret_access_key" {
  value     = aws_iam_access_key.mlflow_user.secret
  sensitive = true
}