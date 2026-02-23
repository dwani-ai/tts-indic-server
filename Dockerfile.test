
# CUDA 12.8 + cuDNN (devel image so you can compile stuff)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive

# Basic OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# (Optional) set default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir --no-deps -r requirements.txt


# Set working directory
WORKDIR /app


# Copy the rest of the application code
COPY . .


# Expose port for the API
EXPOSE 10804


# Run the TTS API server
CMD ["python", "/app/src/server/main.py", "--host", "0.0.0.0", "--port", "10804"]