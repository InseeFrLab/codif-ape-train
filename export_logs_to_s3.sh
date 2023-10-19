mc cp -r s3/projet-ape/log_files/ log_files/
mkdir -p log_files/log_zip
mv -n log_files/raw/*.gz log_files/log_zip/
gunzip log_files/log_zip/*.gz

python ./src/extract_logs.py

mc cp  logs_$(date +'%Y-%m-%d').parquet s3/projet-ape/log_files/preprocessed/logs_$(date +'%Y-%m-%d').parquet
