resource "aws_resourcegroups_group" "f1_predictor_resource_group" {
  name        = "f1-predictor"
  description = "Resources tagged Project f1-predictor"

  resource_query {
    type = "TAG_FILTERS_1_0"
    query = jsonencode({
      ResourceTypeFilters = ["AWS::AllSupported"]
      TagFilters = [{
        Key    = "Project"
        Values = ["f1-predictor"]
      }]
    })
  }
}