"""gRPC inference server for a vision-language model.

Three properties an inference server needs, and where they live:
  - admission control: at most --max-queue requests waiting or running; beyond
    that, reject at once with RESOURCE_EXHAUSTED instead of letting latency grow.
    A request whose client has gone (deadline passed or cancelled) while it waited
    for the GPU is dropped before inference instead of computing an unread answer
  - server-side timing: compute time and GPU-queue time are returned separately
  - graceful degradation: a clear error on a bad image, an unsupported setting or
    GPU out-of-memory, instead of hanging
"""
import argparse, collections, io, threading, time
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
    POLL_S = 0.05            # how often a queued request checks whether its client is gone

    def __init__(self, model_id, max_edge=1536, max_new_tokens=32, device="cuda",
                 prompt="vi", max_queue=8, revision=None):
        from bench.harness import Runner, Config
        self.Config = Config
        self.runner = Runner(model_id, device=device, revision=revision)
        self.default_edge = max_edge
        self.default_tokens = max_new_tokens
        self.device = device
        # reported by Health, so a client can record exactly which weights answered
        self.model_id = f"{model_id}@{revision}" if revision else model_id
        self.prompt = prompt
        # One GPU serves one request at a time. Admitted requests take it strictly in
        # arrival order: a plain lock polled with a timeout lets a newcomer overtake
        # requests that have waited longer, which stretches the latency tail.
        self.gpu_turn = threading.Condition()
        self.gpu_waiting = collections.deque()
        self.gpu_busy = False
        self.admission = Admission(max_queue)

    def Health(self, request, context):
        vram = (torch.cuda.memory_allocated() / 2**20) if self.device == "cuda" else 0.0
        return vlm_pb2.HealthReply(ready=True, model=self.model_id,
                                   device=self.device, vram_mb=vram)

    def _acquire_gpu(self, context):
        """Wait for the GPU, but give up as soon as the client can no longer use the answer.

        A blocking wait would hand the GPU to a request whose deadline passed while it
        queued, spending a full generation on a reply nobody reads.
        """
        ticket = object()
        with self.gpu_turn:
            self.gpu_waiting.append(ticket)
            try:
                while self.gpu_busy or self.gpu_waiting[0] is not ticket:
                    self.gpu_turn.wait(self.POLL_S)
                    remaining = context.time_remaining()
                    if remaining is not None and remaining <= 0:
                        context.abort(grpc.StatusCode.DEADLINE_EXCEEDED,
                                      "deadline passed while queued")
                    if not context.is_active():
                        context.abort(grpc.StatusCode.CANCELLED, "client cancelled while queued")
                if not context.is_active():  # it may have gone just as the GPU freed up
                    context.abort(grpc.StatusCode.CANCELLED, "client gone before inference")
                self.gpu_busy = True
            finally:
                # leaving the line, served or not, lets the next request check its turn
                self.gpu_waiting.remove(ticket)
                self.gpu_turn.notify_all()

    def _release_gpu(self):
        with self.gpu_turn:
            self.gpu_busy = False
            self.gpu_turn.notify_all()

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
            self._acquire_gpu(context)
            try:
                queue_ms = (time.perf_counter() - t_wait) * 1000
                t0 = time.perf_counter()
                try:
                    pred, n_img, _ = self.runner.infer(sample, cfg)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "out of GPU memory")
                server_ms = (time.perf_counter() - t0) * 1000
            finally:
                self._release_gpu()
        finally:
            self.admission.leave()
        return vlm_pb2.InferReply(answer=pred, image_tokens=n_img,
                                  server_ms=server_ms, queue_ms=queue_ms)


def build_server(service, max_queue):
    """One worker per admissible request, plus one so Health answers under full load.

    gRPC hands calls to the executor before any handler runs; with fewer workers than
    the bound, excess calls wait in the executor's unbounded queue where admission
    cannot see them. maximum_concurrent_rpcs makes gRPC itself reject what no worker
    can take, so every accepted call reaches admission at once.
    """
    slots = max_queue + 1
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=slots),
        options=[("grpc.max_receive_message_length", 32 * 1024 * 1024)],
        maximum_concurrent_rpcs=slots)
    vlm_pb2_grpc.add_VlmServiceServicer_to_server(service, server)
    return server


def serve(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from bench.harness import resolve_revision
    revision = resolve_revision(args.model, args.model_revision, "model")
    server = build_server(VlmService(args.model, args.max_edge, args.max_new_tokens, device,
                                     args.prompt, args.max_queue, revision), args.max_queue)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"server ready on port {args.port} | model {args.model} "
          f"| device {device} | max edge {args.max_edge} | prompt {args.prompt} "
          f"| max queue {args.max_queue}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--model-revision", default="main",
                    help="hub branch, tag or commit; resolved to a commit and reported by Health")
    ap.add_argument("--host", default="[::]",
                    help="address to listen on; 127.0.0.1 keeps a local benchmark off the network")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--max-queue", type=int, default=8,
                    help="requests waiting or running before new ones are rejected")
    ap.add_argument("--max-edge", type=int, default=1536, choices=ALLOWED_EDGES)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="vi", choices=("vi", "en", "none"),
                    help="instruction appended to each question; 'vi' is what was measured")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    serve(ap.parse_args())
