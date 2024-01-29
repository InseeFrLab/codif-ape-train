#! /bin/bash
export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr/
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'
export MLFLOW_EXPERIMENT_NAME="test"
ENTRY_POINT=main

Y=apet_finale
DIM=180
LR=0.2
EPOCHS=50
WORDNGRAMS=3
MINN=3
MAXN=4
MINCOUNT=3
BUCKET=2000000
LOSS=ova
PREFIX=__label__
FEATURE1=liasse_type
FEATURE2=activ_nat_et
FEATURE3=activ_surf_et
FEATURE4=evenement_type


mlflow run ~/work/codification-ape/ \
    --env-manager=local \
    --entry-point $ENTRY_POINT \
    -P remote_server_uri=$MLFLOW_TRACKING_URI \
    -P experiment_name=$MLFLOW_EXPERIMENT_NAME \
    -P Y=$Y \
    -P dim=$DIM \
    -P lr=$LR \
    -P epoch=$EPOCHS \
    -P wordNgrams=$WORDNGRAMS \
    -P minn=$MINN \
    -P maxn=$MAXN \
    -P minCount=$MINCOUNT \
    -P bucket=$BUCKET \
    -P loss=$LOSS \
    -P label_prefix=$PREFIX \
    -P categorical_features_1=$FEATURE1 \
    -P categorical_features_2=$FEATURE2 \
    -P categorical_features_3=$FEATURE3 \
    -P categorical_features_4=$FEATURE4
