#!/bin/bash
sudo apt-get update
sudo apt-get install p7zip*

pip install -r requirements.txt
pre-commit install

export PYTHONPATH="$PWD:$PYTHONPATH"
export MC_HOST_minio=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
mc cp -r minio/projet-ape/data/extraction_sirene_20220712_harmonised_20221014.parquet data/
mc cp -r minio/projet-ape/data/extraction_sirene_20220712.parquet data/
mc cp -r minio/projet-ape/data/naf_extended.csv data/
mc cp minio/projet-ape/api_log.7z data/

7z x data/api_log.7z -odata/
rm data/api_log.7z
mv data/CodificationAnalyse.sh.20221005_115614.log data/api_log.log

python - <<'END_SCRIPT'
import nltk
nltk.download('stopwords')
END_SCRIPT
