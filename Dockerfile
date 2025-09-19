
# Use NVIDIA CUDA 12.8.0 with cuDNN development tools on Ubuntu 22.04
FROM slabstech/dwani-vllm


# Set working directory
WORKDIR /app


# Copy the rest of the application code
COPY . .

RUN pip install slowapi pydantic_settings pydub matplotlib vocos torchdiffeq librosa jieba pypinyin x_transformers wandb
RUN pip install ema_pytorch wandb datasets
RUN pip install numpy==1.26.4

# Expose port for the API
EXPOSE 7864


# Run the TTS API server
CMD ["python", "/app/src/gh200/main.py", "--host", "0.0.0.0", "--port", "7864", "--config", "config_two"]