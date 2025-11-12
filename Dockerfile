# Parallel Synth Dataset Generator - Docker Image
# Includes Blender, Python, and all dependencies for 3D rendering at scale

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Metadata
LABEL maintainer="Parallel Synth <contact@parallelsynth.com>"
LABEL description="Parallel Synth 3D Rendering Dataset Generator"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    xz-utils \
    libxrender1 \
    libxi6 \
    libxkbcommon0 \
    libsm6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Blender 4.0
ENV BLENDER_VERSION=4.0.2
ENV BLENDER_URL=https://download.blender.org/release/Blender4.0/blender-${BLENDER_VERSION}-linux-x64.tar.xz

RUN wget -q ${BLENDER_URL} -O blender.tar.xz && \
    tar -xf blender.tar.xz && \
    mv blender-${BLENDER_VERSION}-linux-x64 /opt/blender && \
    rm blender.tar.xz && \
    ln -s /opt/blender/blender /usr/local/bin/blender

# Verify Blender installation
RUN blender --version

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package in development mode
RUN pip3 install -e .

# Create output directories
RUN mkdir -p output/{samples,training_data,reports,logs,cache}

# Set environment variables
ENV BLENDER_PATH=/usr/local/bin/blender
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports (for monitoring dashboard)
EXPOSE 8080 9090

# Default command
CMD ["python3", "examples/generate_first_batch.py"]
