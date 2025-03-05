#!/bin/bash
git config --global credential.helper store

git submodule update --init

# Install dependencies
pip install -e src/models/torch-fastText

pip install -r requirements.txt
pre-commit install

export PYTHONPATH="$PWD:$PYTHONPATH"

AWS_ACCESS_KEY_ID=`vault kv get -field=AWS_ACCESS_KEY_ID onyxia-kv/projet-ape/s3` && export AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=`vault kv get -field=AWS_SECRET_ACCESS_KEY onyxia-kv/projet-ape/s3` && export AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr/

python -m nltk.downloader stopwords
