#! /bin/bash
# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

# Set MLFLOW_TRACKING_URI environment variable
export MLFLOW_TRACKING_URI="https://projet-ape-543061.user.lab.sspcloud.fr"

# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_EXPERIMENT_NAME="Production"

# Set DATA_PATH environment variable
export DATA_PATH="data/data_sirene3.parquet"

# Set CONFIG_PATH environment variable
export CONFIG_PATH="config/config_fasttext27.yaml"


mlflow run ~/work/codification-ape/ --env-manager=local \
-P remote_server_uri=$MLFLOW_TRACKING_URI \
-P experiment_name=$MLFLOW_EXPERIMENT_NAME \
-P data_path=$DATA_PATH \
-P config_path=$CONFIG_PATH
