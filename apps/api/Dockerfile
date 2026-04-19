FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender1 \
	libxcb1 \
	libgl1 \
	&& rm -rf /var/lib/apt/lists/*

COPY apps/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
