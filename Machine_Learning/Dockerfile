FROM python:3.11-slim

# Install system dependencies (libgomp1 is required by faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /code

# Copy requirements and install python packages as root
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a secure non-root user with UID 1000 (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Copy application files and change ownership to user
COPY --chown=user:user . /code

# Switch to the non-root user
USER user

# Set environment variables for HF cache directories & python output
ENV HOME=/home/user \
    TORCH_HOME=/tmp/.cache/torch \
    HF_HOME=/tmp/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Expose port and run the uvicorn API server
CMD ["uvicorn", "ml_api.app:app", "--host", "0.0.0.0", "--port", "7860"]
