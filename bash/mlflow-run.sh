#! /bin/bash
export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr/
export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export MLFLOW_EXPERIMENT_NAME='model_comparison'
ENTRY_POINT=main

# mlflow run ~/work/codif-ape-train/  --env-manager=local -P embedding_dim="40,80" -P num_tokens="1000,2000"
mlflow run ~/work/codif-ape-train/  --env-manager=local
