# Muse API image (CPU). For GPU serving, start from a CUDA base image and install a serving engine instead.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY src/ src/
COPY evals/ evals/
RUN useradd --create-home muse && chown -R muse /app
USER muse
ENV PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "muse.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
