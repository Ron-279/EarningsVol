# Use a slim Python image
FROM python:3.11-slim

# Don't buffer Python output (good for logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system deps (for matplotlib, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (including templates/)
COPY . /app

# Cloud Run will inject $PORT, but default to 8080
ENV PORT=8080

# Use gunicorn to serve the Flask app
# "app:app" = app.py (module) : app (Flask instance)
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 app:app