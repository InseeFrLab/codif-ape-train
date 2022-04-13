#!/bin/bash
pip install -r requirements.txt

export MC_HOST_minio=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY:$AWS_SESSION_TOKEN@$AWS_S3_ENDPOINT
mc cp -r minio/projet-ape/data/ data/
