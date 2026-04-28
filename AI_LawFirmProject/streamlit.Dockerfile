#  streamlit.Dockerfile
FROM python:3.10-slim-bookworm

WORKDIR /app

COPY streamlit_requirements.txt .
RUN pip install --no-cache-dir -r streamlit_requirements.txt

COPY app.py  .

RUN addgroup --system streamlit && adduser --system --ingroup streamlit --home /app streamlit && chown -R streamlit:streamlit /app

USER streamlit

EXPOSE 8501

CMD ["streamlit",  "run",  "app.py",  "--server.port=8501",  "--server.address=0.0.0.0"]
