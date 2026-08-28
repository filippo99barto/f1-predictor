resource "aws_security_group" "rds" {
  name        = "f1-predictor-rds"
  description = "Allow Postgres access from my IP only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Postgres from my IP"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_subnet_group" "default" {
  name       = "f1-predictor-subnet-group"
  subnet_ids = data.aws_subnets.default.ids
}

resource "random_password" "rds_master" {
  length  = 20
  special = false
}

resource "aws_db_instance" "postgres" {
  identifier     = "f1-predictor-db"
  engine         = "postgres"
  engine_version = "16"
  instance_class = "db.t4g.micro"

  allocated_storage = 20
  storage_type      = "gp3"

  username = "postgres"
  password = random_password.rds_master.result

  db_subnet_group_name   = aws_db_subnet_group.default.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = true

  multi_az                = false
  backup_retention_period = 0
  skip_final_snapshot     = true
  deletion_protection     = false
  apply_immediately       = true
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.address
}

output "rds_master_password" {
  value     = random_password.rds_master.result
  sensitive = true
}