
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "video_analytics.api_server", "--database", "/data/events.db", "--host", "0.0.0.0", "--port", "8000"]
