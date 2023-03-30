#!/bin/bash
git config --global credential.helper store

sudo apt-get update
sudo apt-get install p7zip*

pip install -r requirements.txt
pre-commit install

export PYTHONPATH="$PWD:$PYTHONPATH"
#export MC_HOST_minio=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
mc cp s3/projet-ape/data/data_sirene3.parquet data/data_sirene3.parquet
mc cp s3/projet-ape/20230105_logs.zip data/

7z x data/20230105_logs.zip -odata/
rm data/20230105_logs.zip
mv data/concat.log data/api_log.log

python - <<'END_SCRIPT'
import nltk
nltk.download('stopwords')
END_SCRIPT
