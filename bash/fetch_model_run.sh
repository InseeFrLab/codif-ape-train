export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export MLFLOW_TRACKING_URI='https://projet-ape-4549.user.lab.sspcloud.fr/'

root_path="/home/onyxia/work/codification-ape"

python ${root_path}/src/fetch_model.py
