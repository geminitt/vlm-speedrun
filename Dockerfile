# syntax=docker/dockerfile:1
# vlm-speedrun inference server.
#
#   docker build -t vlm-speedrun .
#   docker run --gpus all -p 50051:50051 -v ~/.cache/huggingface:/models vlm-speedrun
#
# The base is plain Ubuntu, not a CUDA image: pip-installed PyTorch ships its own
# CUDA runtime, and the driver comes from the host via nvidia-container-toolkit.
FROM ubuntu:24.04

# gcc and libc6-dev: Triton, which PyTorch uses to generate GPU kernels, compiles a
# small C launcher at runtime. Without them every GPU request fails with "Failed to
# find C compiler" — invisible on CPU, where Triton is never invoked.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gcc libc6-dev && rm -rf /var/lib/apt/lists/*

# pixi builds the environment from the same pixi.lock used in development
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app
COPY pixi.toml pixi.lock ./
# The package cache is shared across builds: changing a layer above does not force
# re-downloading several GB of packages, and an interrupted build can be retried.
# Parallel downloads are cut from the default 50 to 4, since unstable links tend to
# drop when too many connections are open at once.
# UV_HTTP_TIMEOUT: the torch wheel is 529 MB; on a slow link the default timeout of
# about 30 seconds made the installer abandon the whole file partway through.
RUN --mount=type=cache,target=/root/.cache/rattler \
    UV_HTTP_TIMEOUT=900 pixi install --locked -e default --concurrent-downloads 4

COPY bench/ bench/
COPY serve/ serve/

# The gRPC code is not in git; generate it into the image so runtime needs no write access
RUN pixi run proto

# Model weights download at runtime; mount the host cache here to avoid fetching
# 4.5 GB on every container start
ENV HF_HOME=/models
# Unbuffered stdout: otherwise the "server ready" line sits in the buffer and
# `docker logs` makes a healthy server look like it has not started
ENV PYTHONUNBUFFERED=1
EXPOSE 50051

# Default to the edge-768 configuration: twice the baseline throughput
CMD ["pixi", "run", "python", "-m", "serve.server", "--max-edge", "768"]
