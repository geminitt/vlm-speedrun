# Máy chủ suy luận vlm-speedrun.
#
#   docker build -t vlm-speedrun .
#   docker run --gpus all -p 50051:50051 -v ~/.cache/huggingface:/models vlm-speedrun
#
# Ảnh nền là Ubuntu thường, không phải ảnh CUDA: bản PyTorch cài qua pip đã mang
# theo thư viện CUDA runtime, còn driver lấy từ máy chủ qua nvidia-container-toolkit.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && rm -rf /var/lib/apt/lists/*

# pixi dựng môi trường từ đúng pixi.lock dùng lúc phát triển
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install --locked -e default

COPY bench/ bench/
COPY serve/ serve/

# Mã gRPC không nằm trong git; sinh sẵn vào ảnh để lúc chạy không cần quyền ghi
RUN pixi run proto

# Trọng số mô hình tải về lúc chạy; gắn thư mục cache của máy chủ vào đây để khỏi
# tải lại 4,5 GB mỗi lần khởi động container
ENV HF_HOME=/models
# Không đệm stdout: nếu không, dòng "máy chủ sẵn sàng" kẹt trong bộ đệm và
# docker logs trông như máy chủ chưa khởi động xong
ENV PYTHONUNBUFFERED=1
EXPOSE 50051

# Mặc định chạy cấu hình cạnh dài 768: thông lượng gấp đôi bản gốc
CMD ["pixi", "run", "python", "-m", "serve.server", "--max-edge", "768"]
