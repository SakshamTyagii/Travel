FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Make sure the model and data files are available
ENV MODEL_PATH=/app
ENV DATA_DIR=/app

EXPOSE 8080

# This environment variable is used by the gunicorn server
ENV PORT=8080

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 search:app
