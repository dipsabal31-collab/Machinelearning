FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/tmp/hf PORT=7860
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "app:app"]
