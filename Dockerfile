FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project sources
COPY src/ ./src/
COPY data/ ./data/

# Expose nothing by default — this image is for batch scripts, not a server
# Override CMD at runtime, e.g.:
#   docker run aml-risk python -m src.preprocessing
#   docker run aml-risk python -m src.train
CMD ["python", "-m", "src.train"]
