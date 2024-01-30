FROM inseefrlab/onyxia-vscode-pytorch:py3.11.6

RUN git clone https://github.com/InseeFrLab/codif-ape-train.git &&\
    cd codif-ape-train/ &&\
    python - <<'END_SCRIPT' \
    import nltk \
    nltk.download('stopwords') \
    END_SCRIPT

WORKDIR ${WORKSPACE_DIR}/codif-ape-train/
