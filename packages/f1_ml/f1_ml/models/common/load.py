import mlflow.sklearn


def load_model(model_name: str):
    return mlflow.sklearn.load_model(f"models:/{model_name}/latest")
