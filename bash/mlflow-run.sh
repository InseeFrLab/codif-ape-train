#! /bin/bash
export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr/
export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export MLFLOW_EXPERIMENT_NAME=Test
ENTRY_POINT=main

# Parameters
DIM=180
LR=5e-5
EPOCHS=1
WORDNGRAMS=3
MINN=3
MAXN=4
MINCOUNT=3
BUCKET=2000000
LOSS=ova

# Database arguments
Y=apet_finale
PREFIX=__label__
TEXT_FEATURE=libelle_activite_apet
FEATURE1=liasse_type
FEATURE2=activ_nat_et
FEATURE3=activ_surf_et
FEATURE4=evenement_type
FEATURE5=cj
MODEL_CLASS=camembert
DIM1=3
DIM2=3
DIM3=1
DIM4=3
DIM5=3
PRE_TRAINING_WEIGHTS=camembert/camembert-base-ccnet

mlflow run ~/work/codif-ape-train/ \
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
    -P text_feature=$TEXT_FEATURE \
    -P categorical_features_1=$FEATURE1 \
    -P categorical_features_2=$FEATURE2 \
    -P categorical_features_3=$FEATURE3 \
    -P categorical_features_4=$FEATURE4 \
    -P categorical_features_5=$FEATURE5 \
    -P embedding_dim_1=$DIM1 \
    -P embedding_dim_2=$DIM2 \
    -P embedding_dim_3=$DIM3 \
    -P embedding_dim_4=$DIM4 \
    -P embedding_dim_5=$DIM5 \
    -P model_class=$MODEL_CLASS
