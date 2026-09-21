FROM python:3.11-slim

WORKDIR /app

# Without this, print() is block-buffered when stdout isn't a tty, so startup
# diagnostics (including the embedding-model mismatch warning) sit invisible in a
# buffer until some unrelated request happens to flush them.
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi==0.115.0 "uvicorn[standard]==0.30.6" httpx==0.27.2 pydantic==2.9.2 python-dotenv==1.0.1

COPY memory_api.py /app/memory_api.py
COPY profile_schema.py profile_fields.json episode.py runtime_config.py /app/

EXPOSE 8100

CMD ["uvicorn", "memory_api:app", "--host", "0.0.0.0", "--port", "8100"]
