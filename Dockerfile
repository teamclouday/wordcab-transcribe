################################
# BASE
# Sets up all our shared environment variables
################################
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=2.1.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV="/venv" \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Install Python and required build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    curl \
    ffmpeg \
    && apt-get clean

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV/bin:$PATH"

# Create venv and install poetry
RUN python -m venv $VIRTUAL_ENV

# Set work directory
WORKDIR /app
ENV PYTHONPATH="/app:$PYTHONPATH"

################################
# BUILDER
# Used to build deps + create our virtual environment
################################
FROM base AS builder

# Install poetry
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python -

# Install dependencies only
WORKDIR /app
COPY poetry.lock pyproject.toml ./

# Install runtime dependencies with caching
RUN --mount=type=cache,target=/root/.cache/pip \
    poetry install --no-root --only main

################################
# PRODUCTION
# Final image used for runtime
################################
FROM base AS final

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="/venv" \
    PYTHONPATH="/app:$PYTHONPATH"

# Copy venv from builder stage
COPY --from=builder $POETRY_HOME $POETRY_HOME
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV

# Add venv to path
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy application code
WORKDIR /app
COPY poetry.lock pyproject.toml ./
COPY app/ app/
COPY .env.production .env

# Run the application
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]