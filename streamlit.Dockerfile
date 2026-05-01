# Tegifa Legal — Streamlit Application Container
FROM python:3.10-slim-bookworm

WORKDIR /app

# Create non-root user
RUN adduser --system --group --home /home/tegifa tegifa && \
    chown -R tegifa:tegifa /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY app.py .
COPY config.yaml .
COPY db/ ./db/
COPY agents/ ./agents/
COPY rag_scripts/ ./rag_scripts/

RUN chown -R tegifa:tegifa /app

USER tegifa

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
