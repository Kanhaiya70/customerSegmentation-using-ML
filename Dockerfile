# Use official Python slim image
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y wkhtmltopdf && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/segment_classifier.pkl backend/

# Expose Flask port
EXPOSE 5000

# Run your app
CMD ["python", "backend/app.py"]