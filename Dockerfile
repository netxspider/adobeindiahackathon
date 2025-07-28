# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system deps (including for PyMuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcrypt1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

WORKDIR /app

# Install minimal deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcrypt1 \
    && rm -rf /var/lib/apt/lists/*

# Copy packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy script
COPY main.py .

# Create writable directories for input/output and Hugging Face cache
RUN mkdir -p /app/input /app/output /app/cache/huggingface

# Run the script
CMD ["python", "main.py"]
