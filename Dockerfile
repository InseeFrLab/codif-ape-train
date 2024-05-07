FROM inseefrlab/onyxia-vscode-pytorch:py3.11.9

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    python -m nltk.downloader stopwords
