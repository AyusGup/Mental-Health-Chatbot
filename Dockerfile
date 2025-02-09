# Use Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure /tmp is writable
RUN mkdir -p /tmp && chmod -R 777 /tmp

# Expose FastAPI's default port
EXPOSE 7860

# Set Hugging Face cache directory
ENV TRANSFORMERS_CACHE=/tmp

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
