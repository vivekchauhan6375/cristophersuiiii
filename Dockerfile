FROM python:3.10.14-bookworm

RUN pip install --upgrade pip

COPY src /app/src

COPY entrypoint.sh /app/entrypoint.sh

WORKDIR /app

RUN mkdir -p /app/src/trained_models

RUN chmod -R 777 /app/src

RUN chmod +x /app/entrypoint.sh

RUN pip install -r /app/src/requirements.txt

ENV PYTHONPATH=${PYTHONPATH}:/app/src

ENTRYPOINT ["/app/ENTRYPOINT.sh"]