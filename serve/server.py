"""gRPC inference server for a vision-language model.

Three properties an inference server needs, and where they live:
  - admission control: at most --max-queue requests waiting or running; beyond
    that, reject at once with RESOURCE_EXHAUSTED instead of letting latency grow
  - server-side timing: compute time and GPU-queue time are returned separately
  - graceful degradation: a clear error on a bad image, an unsupported setting or
    GPU out-of-memory, instead of hanging
"""
import argparse, io, threading, time
from concurrent import futures

import grpc
import torch
from PIL import Image

from serve.admission import Admission
from serve.gen_proto import ensure_stubs
ensure_stubs()                      # a fresh clone has no code generated from .proto yet
from serve import vlm_pb2, vlm_pb2_grpc  # noqa: E402

# Longest-edge settings the server accepts. Each one keeps its own image processor,
# so arbitrary values would grow memory without bound.
ALLOWED_EDGES = (768, 1152, 1536)


class VlmService(vlm_pb2_grpc.VlmServiceServicer):
    def __init__(self, model_id, max_edge=1536, max_new_tokens=32, device="cuda",
                 prompt="vi", max_queue=8):
        from bench.harness import Runner, Config
        self.Config = Config
        self.runner = Runner(model_id, device=device)
        self.default_edge = max_edge
        self.default_tokens = max_new_tokens
        self.device = device
        self.model_id = model_id
        self.prompt = prompt
        # One GPU serves one request at a time; the lock makes admitted requests
        # queue in order instead of competing for memory.
        self.gpu = threading.Lock()
        self.admission = Admission(max_queue)

    def Health(self, request, context):
        vram = (torch.cuda.memory_allocated() / 2**20) if self.device == "cuda" else 0.0
        return vlm_pb2.HealthReply(ready=True, model=self.model_id,
                                   device=self.device, vram_mb=vram)

    def Infer(self, request, context):
        edge = request.max_edge or self.default_edge
        if edge not in ALLOWED_EDGES:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"max_edge must be one of {ALLOWED_EDGES}, got {edge}")
        try:
            img = Image.open(io.BytesIO(request.image)).convert("RGB")
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"unreadable image: {e}")
        if not self.admission.try_enter():
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                          f"server busy: {self.admission.max_in_flight} requests in flight")
        try:
            cfg = self.Config("serve", max_edge=edge, prompt=self.prompt,
                              max_new_tokens=request.max_new_tokens or self.default_tokens)
            sample = {"image": img, "query": request.question or "Describe the image."}
            t_wait = time.perf_counter()
            with self.gpu:
                queue_ms = (time.perf_counter() - t_wait) * 1000
                t0 = time.perf_counter()
                try:
                    pred, n_img, _ = self.runner.infer(sample, cfg)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "out of GPU memory")
                server_ms = (time.perf_counter() - t0) * 1000
        finally:
            self.admission.leave()
        return vlm_pb2.InferReply(answer=pred, image_tokens=n_img,
                                  server_ms=server_ms, queue_ms=queue_ms)


def serve(args):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_queue),
        options=[("grpc.max_receive_message_length", 32 * 1024 * 1024)])
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm_pb2_grpc.add_VlmServiceServicer_to_server(
        VlmService(args.model, args.max_edge, args.max_new_tokens, device,
                   args.prompt, args.max_queue), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"server ready on port {args.port} | model {args.model} "
          f"| device {device} | max edge {args.max_edge} | prompt {args.prompt} "
          f"| max queue {args.max_queue}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--max-queue", type=int, default=8,
                    help="requests waiting or running before new ones are rejected")
    ap.add_argument("--max-edge", type=int, default=1536, choices=ALLOWED_EDGES)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="vi", choices=("vi", "en", "none"),
                    help="instruction appended to each question; 'vi' is what was measured")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    serve(ap.parse_args())
