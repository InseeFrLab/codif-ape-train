FROM inseefrlab/onyxia-vscode-pytorch:py3.12.6

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    python -m nltk.downloader stopwords
