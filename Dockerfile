
FROM python:3.12-slim

WORKDIR /app

# Install application dependencies, including deep-learning models
COPY requirements.txt requirements-deep.txt requirements-prod.txt ./

RUN pip install --no-cache-dir -r requirements-prod.txt \
    && pip install --no-cache-dir -r requirements-deep.txt

# Copy application source and existing trained models
COPY scamshield ./scamshield
COPY models ./models

# Create writable storage for database and feedback
RUN mkdir -p /app/data \
    && chown -R nobody:nogroup /app/data

ENV SCAMSHIELD_HOST=0.0.0.0
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

USER nobody

CMD ["python", "-m", "scamshield.app"]
