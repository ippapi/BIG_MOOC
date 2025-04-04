# Use the python 3.9-slim image
FROM python:3.9-slim

# Install required dependencies like curl and tar
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application code
COPY . /app

# Set working directory
WORKDIR /app

# Command to run the app
CMD ["bash", "-c", "echo 'Waiting for Cassandra to start...' && until python3 /app/data_loader.py; do echo 'Keyspace not ready yet... waiting 5s'; sleep 5; done"]

