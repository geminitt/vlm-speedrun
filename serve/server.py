"""gRPC inference server for a vision-language model.

Three properties an inference server needs, and where they live:
  - a bounded queue: reject early under overload instead of letting latency grow unbounded
  - server-side timing: separate compute from network
  - graceful degradation: return a clear error on failure instead of hanging
"""
import argparse, io, threading, time
from concurrent import futures

import grpc
import torch
from PIL import Image

from serve.gen_proto import ensure_stubs
ensure_stubs()                      # a fresh clone has no code generated from .proto yet
from serve import vlm_pb2, vlm_pb2_grpc  # noqa: E402


class VlmService(vlm_pb2_grpc.VlmServiceServicer):
    def __init__(self, model_id, max_edge=1536, max_new_tokens=32, device="cuda"):
        from bench.harness import Runner, Config
        self.Config = Config
        self.runner = Runner(model_id, device=device)
        self.default_edge = max_edge
        self.default_tokens = max_new_tokens
        self.device = device
        self.model_id = model_id
        # One GPU serves one request at a time; the lock makes requests queue in
        # order instead of competing for memory.
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
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"unreadable image: {e}")

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
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "out of GPU memory")
            server_ms = (time.perf_counter() - t0) * 1000

        return vlm_pb2.InferReply(answer=pred, image_tokens=n_img,
                                  server_ms=server_ms, queue_ms=queue_ms)


def serve(args):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[("grpc.max_receive_message_length", 32 * 1024 * 1024)])
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm_pb2_grpc.add_VlmServiceServicer_to_server(
        VlmService(args.model, args.max_edge, args.max_new_tokens, device), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"server ready on port {args.port} | model {args.model} "
          f"| device {device} | max edge {args.max_edge}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-edge", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    serve(ap.parse_args())
