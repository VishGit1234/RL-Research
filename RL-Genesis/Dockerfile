# Use Ubuntu 22.04 as base
FROM ubuntu:22.04

# Set a working directory
WORKDIR /src

# Install necessary tools (based on genesis-sim dockerfile)
# We use --no-cache-dir to avoid caching package data, reducing image size
# We also clean up apt lists afterwards to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # build-essential \
    # libvulkan-dev \
    # zlib1g-dev \
    # xorg-dev libglu1-mesa-dev \
    # libnvidia-gl-525 \
    # libsnappy-dev \
    # software-properties-common \
    libnvidia-gl-550 \
    libgl1 \
    libx11-dev \
    libglew-dev \
    libxrender1 \
    libglib2.0-0 \
    wget \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Define the URL for the OMPL wheel file for Python 3.10 on manylinux_2_28_x86_64
# NOTE: Replace this URL with the actual download URL of the wheel file.
# This is a placeholder URL. You need to find the correct URL for the wheel.
ARG OMPL_WHEEL="ompl-1.7.0-cp310-cp310-manylinux_2_28_x86_64.whl"
ARG OMPL_WHEEL_URL="https://github.com/ompl/ompl/releases/download/prerelease/${OMPL_WHEEL}"

# Download the wheel file
RUN wget -O /tmp/${OMPL_WHEEL} ${OMPL_WHEEL_URL}

# Install the OMPL wheel file using pip (as well as torch and genesis)
# Remove the wheel file afterwards
RUN pip install --no-cache-dir torch genesis-world /tmp/${OMPL_WHEEL} && \
    rm /tmp/${OMPL_WHEEL}

# You can add further instructions here to set up your application
# For example, copy your application code into the container:
COPY . /src

# Set the default command to run when the container starts (optional)
# CMD ["python3"]


