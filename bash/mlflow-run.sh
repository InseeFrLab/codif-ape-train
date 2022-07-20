#! /bin/bash
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

# Set MLFLOW_TRACKING_URI environment variable
GET_PODS=`kubectl get pods`

while IFS= read -r line; do
    VAR=`echo "${line}" | sed -n "s/.*mlflow-\([0-9]\+\)-.*/\1/p"`
    if [ -z "$VAR" ]; then
        :
    else
        POD_ID=$VAR
    fi
done <<< "$GET_PODS"

export MLFLOW_TRACKING_URI="https://projet-ape-$POD_ID.kub.sspcloud.fr"
export MLFLOW_EXPERIMENT_NAME="test"

mlflow run ~/work/codification-ape/ --env-manager=local -P remote_server_uri=$MLFLOW_TRACKING_URI -P data_path=data/extraction_sirene_20220712_harmonised.parquet -P config_path=config/config_fasttext27.yaml
mlflow run ~/work/codification-ape/ --env-manager=local -P remote_server_uri=$MLFLOW_TRACKING_URI -P data_path=data/extraction_sirene_20220712_harmonised.parquet -P config_path=config/config_fasttext33.yaml
