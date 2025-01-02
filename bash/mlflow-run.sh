#! /bin/bash
export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr/
export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export MLFLOW_EXPERIMENT_NAME='Experimentation'
ENTRY_POINT=main

# Parameters
DIM=180
LR=0.2
EPOCHS=50
WORDNGRAMS=3
MINN=3
MAXN=4
MINCOUNT=3
BUCKET=2000000
LOSS=ova

# Database arguments
#Y=apet_finale
PREFIX=__label__
TEXT_FEATURE=libelle
TEXTUAL_FEATURE1=activ_nat_lib_et
TEXTUAL_FEATURE2=activ_sec_agri_et
FEATURE1=TYP
FEATURE2=NAT
FEATURE3=SRF
FEATURE4=EVT
FEATURE5=CJ
FEATURE6=CRT
MODEL_CLASS=fasttext
# DIM1=1
# DIM2=3
# DIM3=3
# DIM4=1
# DIM5=3
# DIM6=1
# PRE_TRAINING_WEIGHTS=camembert/camembert-base
START_YEAR=2025

mlflow run ~/work/codif-ape-train/ \
    --env-manager=local \
    --entry-point $ENTRY_POINT \
    -P remote_server_uri=$MLFLOW_TRACKING_URI \
    -P revision=$revision \
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
    -P textual_features_1=$TEXTUAL_FEATURE1 \
    -P textual_features_2=$TEXTUAL_FEATURE2 \
    -P categorical_features_1=$FEATURE1 \
    -P categorical_features_2=$FEATURE2 \
    -P categorical_features_3=$FEATURE3 \
    -P categorical_features_4=$FEATURE4 \
    -P categorical_features_5=$FEATURE5 \
    -P categorical_features_6=$FEATURE6 \
    -P model_class=$MODEL_CLASS \
    -P start_year=$START_YEAR

    # -P embedding_dim_1=$DIM1 \
    # -P embedding_dim_2=$DIM2 \
    # -P embedding_dim_3=$DIM3 \
    # -P embedding_dim_4=$DIM4 \
    # -P embedding_dim_5=$DIM5 \
    # -P embedding_dim_6=$DIM6 \
    # -P pre_training_weights=$PRE_TRAINING_WEIGHTS \
    # -P experiment_name=$MLFLOW_EXPERIMENT_NAME \
    # -P Y=$Y \
