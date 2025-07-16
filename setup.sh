#!/bin/bash

uv sync
uv run pre-commit install

export PYTHONPATH="$PWD:$PYTHONPATH"

export AWS_SESSION_TOKEN=""
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr/
