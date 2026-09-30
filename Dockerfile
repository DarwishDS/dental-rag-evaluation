FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data
ENV DENTAL_CACHE_DIR=/app/.cache
EXPOSE 8000
CMD ["dental-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
