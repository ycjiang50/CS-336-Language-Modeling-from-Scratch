# Use Python 3.11 slim image
FROM python:3.11-slim-bookworm

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install system dependencies
# git is often needed for pip installing from git repos
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first to leverage cache
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --frozen ensures we use the exact versions in uv.lock
# --system installs into the system python, avoiding venv complexity in Docker
RUN uv sync --frozen --system

# Copy the rest of the application
COPY . .

# Set PYTHONPATH so python can find the modules
ENV PYTHONPATH=/app

# Default entrypoint (can be overridden in k8s manifest)
ENTRYPOINT ["python"]
CMD ["cs336_basics/train.py", "--help"]
