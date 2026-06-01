# Use Python 3.11 slim base image
FROM python:3.11-slim

# Optional build argument to control which YOLO package/version to install.
# If you have a package name or pinned version (for example: "ultralytics==11.0.0"),
# pass it at build time: docker build --build-arg YOLO_PACKAGE="ultralytics==11.0.0" -t myapp:latest .
# Default is empty so a normal build won't attempt to download large vision packages.
ARG YOLO_PACKAGE=""

WORKDIR /app

# Install system dependencies required by common vision packages (ffmpeg, libgl) and build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       ffmpeg \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Try to install the YOLO package specified by the build-arg. If it fails the build will continue
# (so the container can still run the Flask app). If you need a failing build instead, remove the
# `|| true` so that `pip install` failing will fail the docker build.
RUN if [ "${YOLO_PACKAGE}" != "" ]; then \
      pip install --no-cache-dir ${YOLO_PACKAGE} || true; \
    fi

# Copy the application code
COPY . /app

# Expose the port used by the Flask app
EXPOSE 5001

# Default command: run the Flask app directly. For production consider using gunicorn.
CMD ["python", "app.py"]
