#! /bin/bash
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

mlflow models serve --model-uri $MODEL_URI -h 0.0.0.0 -p $SERVING_PORT --no-conda
