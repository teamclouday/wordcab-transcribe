################################
# BUILDER
# Used to build deps + create our virtual environment
################################
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV="/venv"

# Install Python and required build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Create venv and install poetry
RUN python -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install "poetry==$POETRY_VERSION"

# Add venv to path
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

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
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS final

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="/venv" \
    PYTHONPATH="/app:$PYTHONPATH"

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Copy venv from builder stage
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV

# Copy application code
WORKDIR /app
COPY app/ app/
COPY .env.production .env

# Run the application
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]