# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python script and CSV file to the container
COPY bot.py .
COPY main.csv .

# Set environment variables (optional, adjust as needed)
ENV PYTHONUNBUFFERED=1

# Run the bot when the container launches
CMD ["python", "bot.py"]