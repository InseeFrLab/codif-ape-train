#! /bin/bash
# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

# Set MLFLOW_TRACKING_URI environment variable
export MLFLOW_TRACKING_URI="https://projet-ape-537858.user.lab.sspcloud.fr"

# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_EXPERIMENT_NAME="test"

mlflow run ~/work/codification-ape/ --env-manager=local -P remote_server_uri=$MLFLOW_TRACKING_URI -P data_path=data/extraction_sirene_20220712_harmonized_20221014.parquet -P config_path=config/config_fasttext27.yaml
