FROM inseefrlab/onyxia-vscode-pytorch:py3.11.6

RUN git clone https://github.com/InseeFrLab/codif-ape-train.git &&\
    cd codif-ape-train/ &&\
    pip install --no-cache-dir --upgrade -r requirements.txt && \
    python -m nltk.downloader stopwords

WORKDIR ${WORKSPACE_DIR}/codif-ape-train/
