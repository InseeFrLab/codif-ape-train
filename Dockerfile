FROM python:3.10

COPY serving_requirements.txt /serving_requirements.txt

RUN pip install -r serving_requirements.txt

COPY bash/serving_entrypoint.sh /serving_entrypoint.sh

ENTRYPOINT [ "/bin/bash", "/serving_entrypoint.sh" ]