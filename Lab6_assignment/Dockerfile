# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libstdc++6 \
    libnss3 \
    libc6 \
    libgssapi-krb5-2 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . . 

# Cloud Run requires port 8080
EXPOSE 8080

# Start FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
