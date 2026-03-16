FROM python:3.11-slim

WORKDIR /app

COPY models/ ./models/

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app/main.py"]