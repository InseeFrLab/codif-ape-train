#!/bin/bash
pip install -r requirements.txt
pre-commit install

export PYTHONPATH="$PWD:$PYTHONPATH"
export MC_HOST_minio=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
mc cp -r minio/projet-ape/data/extraction_sirene_20220712_harmonised.parquet data/
mc cp -r minio/projet-ape/data/extraction_sirene_20220712.parquet data/
mc cp -r minio/projet-ape/data/naf_extended.csv data/

python - <<'END_SCRIPT'
import nltk
nltk.download('stopwords')
END_SCRIPT
