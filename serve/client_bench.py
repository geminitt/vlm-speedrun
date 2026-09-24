"""Measure the server's end-to-end latency at several concurrency levels.

The question: when several users call at once on ONE GPU, how much does p95 tail
latency grow, and does throughput increase at all?
"""
import argparse, io, json, statistics, time
from concurrent import futures
from pathlib import Path

import grpc

from bench.harness import load_samples
from bench.latency_probe import summarize
from serve.gen_proto import ensure_stubs
ensure_stubs()                      # a fresh clone has no code generated from .proto yet
from serve import vlm_pb2, vlm_pb2_grpc  # noqa: E402


def encode(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def one_call(stub, payload, question, max_edge, timeout):
    t0 = time.perf_counter()
    reply = stub.Infer(vlm_pb2.InferRequest(
        image=payload, question=question, max_edge=max_edge), timeout=timeout)
    return (time.perf_counter() - t0) * 1000, reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="localhost:50051")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--max-edge", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="results/gate4_serving.json")
    a = ap.parse_args()

    ch = grpc.insecure_channel(a.target, options=[
        ("grpc.max_send_message_length", 32 * 1024 * 1024)])
    stub = vlm_pb2_grpc.VlmServiceStub(ch)
    h = stub.Health(vlm_pb2.HealthRequest(), timeout=60)
    print(f"server: {h.model} on {h.device}")

    samples = load_samples(a.samples, seed=1)
    payloads = [(encode(s["image"]), s["query"]) for s in samples]

    # warm up
    one_call(stub, *payloads[0], a.max_edge, a.timeout)

    res = {}
    for c in [int(x) for x in a.concurrency.split(",")]:
        lat, srv, t0 = [], [], time.perf_counter()
        with futures.ThreadPoolExecutor(max_workers=c) as ex:
            jobs = [ex.submit(one_call, stub, p, q, a.max_edge, a.timeout)
                    for p, q in payloads]
            for j in jobs:
                ms, reply = j.result()
                lat.append(ms); srv.append(reply.server_ms)
        wall = time.perf_counter() - t0
        res[str(c)] = {"end_to_end_ms": summarize(lat),
                       "server_ms": summarize(srv),
                       "throughput_rps": len(lat) / wall}
        e = res[str(c)]["end_to_end_ms"]
        print(f"concurrency {c}: median {e['median']:.0f} ms | p95 {e['p95']:.0f} ms "
              f"| throughput {res[str(c)]['throughput_rps']:.2f} req/s")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
