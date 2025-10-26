FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY prometheus_exporter.py .
CMD ["python", "prometheus_exporter.py"]