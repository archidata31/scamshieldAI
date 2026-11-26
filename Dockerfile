FROM python:3.12-slim
WORKDIR /app
COPY requirements-prod.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY scamshield ./scamshield
RUN python -m scamshield.data_gen && python -m scamshield.train --no-real
ENV SCAMSHIELD_HOST=0.0.0.0 PORT=5000
EXPOSE 5000
USER nobody
CMD ["python", "-m", "scamshield.app"]
