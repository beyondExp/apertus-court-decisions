FROM python:3.11-slim-bookworm

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src ./src
COPY data ./data
COPY checkpoint ./checkpoint

ENV PYTHONUNBUFFERED=1
CMD ["python", "src/run.py"]
