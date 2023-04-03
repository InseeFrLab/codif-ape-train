# export MLFLOW_TRACKING_URI="https://projet-ape-4549.user.lab.sspcloud.fr"
# export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

import sys

import mlflow

sys.path.append("./src/")

model_name = "FastText-APE"
version = 3
stage = "Production"

mlflow.set_tracking_uri("https://projet-ape-4549.user.lab.sspcloud.fr")

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version}")
