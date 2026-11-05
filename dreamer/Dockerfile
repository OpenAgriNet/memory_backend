# Build from the repo root: docker build -f dreamer/Dockerfile .
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir openai==1.109.1 requests==2.32.5 python-dotenv==1.0.1

COPY service/runtime_config.py service/profile_schema.py service/profile_fields.json service/episode.py /app/service/
COPY dreamer/ /app/dreamer/

WORKDIR /app/dreamer
CMD ["python", "worker.py"]
