# Ảnh chứa máy chủ suy luận. Cần GPU NVIDIA lúc chạy (--gpus all).
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git && rm -rf /var/lib/apt/lists/*

# pixi quản lý môi trường, giống hệt lúc phát triển, nên bản trong ảnh và bản
# trên máy dùng cùng một pixi.lock
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install --locked

COPY bench/ bench/
COPY serve/ serve/
COPY tests/ tests/

ENV HF_HOME=/models
EXPOSE 50051

# Mặc định chạy cấu hình nhanh: cạnh dài 768 cho thông lượng gấp đôi
CMD ["pixi", "run", "python", "-m", "serve.server", "--max-edge", "768"]
