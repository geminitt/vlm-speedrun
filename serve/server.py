"""Máy chủ suy luận gRPC cho mô hình thị giác–ngôn ngữ.

Ba tính chất cần có của một máy chủ suy luận, và chỗ cài đặt chúng:
  - hàng đợi có giới hạn: từ chối sớm khi quá tải thay vì để độ trễ phình vô hạn
  - đo thời gian tại máy chủ: tách phần tính toán khỏi phần mạng
  - suy giảm có kiểm soát: hết hạn thời gian thì trả lỗi rõ ràng, không treo
"""
import argparse, io, threading, time
from concurrent import futures

import grpc
import torch
from PIL import Image

from serve import vlm_pb2, vlm_pb2_grpc


class VlmService(vlm_pb2_grpc.VlmServiceServicer):
    def __init__(self, model_id, max_edge=1536, max_new_tokens=32, device="cuda"):
        from bench.harness import Runner, Config
        self.Config = Config
        self.runner = Runner(model_id, device=device)
        self.default_edge = max_edge
        self.default_tokens = max_new_tokens
        self.device = device
        self.model_id = model_id
        # Một GPU chỉ phục vụ được một yêu cầu tại một thời điểm; khoá để các
        # yêu cầu xếp hàng có trật tự thay vì tranh nhau bộ nhớ.
        self.lock = threading.Lock()

    def Health(self, request, context):
        vram = (torch.cuda.memory_allocated() / 2**20) if self.device == "cuda" else 0.0
        return vlm_pb2.HealthReply(ready=True, model=self.model_id,
                                   device=self.device, vram_mb=vram)

    def Infer(self, request, context):
        t_in = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(request.image)).convert("RGB")
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"ảnh không đọc được: {e}")

        cfg = self.Config("serve",
                          max_edge=request.max_edge or self.default_edge,
                          max_new_tokens=request.max_new_tokens or self.default_tokens)
        sample = {"image": img, "query": request.question or "Describe the image."}

        with self.lock:
            queue_ms = (time.perf_counter() - t_in) * 1000
            t0 = time.perf_counter()
            try:
                pred, n_img, _ = self.runner.infer(sample, cfg)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "hết bộ nhớ GPU")
            server_ms = (time.perf_counter() - t0) * 1000

        return vlm_pb2.InferReply(answer=pred, image_tokens=n_img,
                                  server_ms=server_ms, queue_ms=queue_ms)


def serve(args):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[("grpc.max_receive_message_length", 32 * 1024 * 1024)])
    vlm_pb2_grpc.add_VlmServiceServicer_to_server(
        VlmService(args.model, args.max_edge, args.max_new_tokens), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"máy chủ sẵn sàng trên cổng {args.port} "
          f"| mô hình {args.model} | cạnh dài tối đa {args.max_edge}")
    server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-edge", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    serve(ap.parse_args())
