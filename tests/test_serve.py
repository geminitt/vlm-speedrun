"""Overload and deadline behavior of the gRPC server, over a real localhost channel.

The model is replaced by a fake runner that blocks until the test releases it, so
the protocol, transport, executor and service handler are the real ones and the
timing of each scenario is controlled by the test, not by sleeps.
"""
import io
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
grpc = pytest.importorskip("grpc")
from PIL import Image  # noqa: E402

import bench.harness as harness  # noqa: E402


class FakeRunner:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def infer(self, sample, cfg):
        self.calls.append(sample["query"])
        self.started.set()
        assert self.release.wait(10), "the test never released the fake model"
        return "ok", 81, 100


@pytest.fixture
def serve(monkeypatch):
    """Start the server on a free localhost port; yield (service, stub)."""
    monkeypatch.setattr(harness, "Runner", FakeRunner)
    from serve.server import VlmService, build_server, vlm_pb2_grpc
    started = []

    def start(max_queue):
        service = VlmService("fake", device="cpu", max_queue=max_queue)
        server = build_server(service, max_queue)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(channel).result(timeout=5)
        started.append((server, channel, service))
        return service, vlm_pb2_grpc.VlmServiceStub(channel)

    yield start
    for server, channel, service in started:
        service.runner.release.set()
        channel.close()
        server.stop(0).wait(timeout=5)


def _request(question):
    from serve.server import vlm_pb2
    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    return vlm_pb2.InferRequest(image=buf.getvalue(), question=question)


def _status(call):
    try:
        call.result(timeout=10)
        return "OK"
    except grpc.RpcError as e:
        return e.code().name


@pytest.mark.parametrize("max_queue", [1, 2])
def test_requests_beyond_the_bound_are_rejected_while_admitted_ones_wait(serve, max_queue):
    service, stub = serve(max_queue)
    first = stub.Infer.future(_request("q0"), timeout=10)
    assert service.runner.started.wait(5)
    rest = [stub.Infer.future(_request(f"q{i}"), timeout=10) for i in range(1, max_queue + 3)]
    # The model is still blocked: every rejection must arrive before any admitted
    # request can finish, otherwise excess calls are queueing instead of being refused.
    rejected = [c for c in rest if _status_if_done(c, 3) == "RESOURCE_EXHAUSTED"]
    assert len(rejected) == 3
    assert not first.done()
    service.runner.release.set()
    statuses = [_status(c) for c in [first] + rest]
    assert statuses.count("OK") == max_queue
    assert statuses.count("RESOURCE_EXHAUSTED") == 3


def _status_if_done(call, wait_s):
    """Status of a call that finishes within wait_s, or None if it is still pending."""
    deadline = time.monotonic() + wait_s
    while not call.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    return _status(call) if call.done() else None


def test_health_answers_while_the_queue_is_full(serve):
    from serve.server import vlm_pb2
    service, stub = serve(1)
    first = stub.Infer.future(_request("q0"), timeout=10)
    assert service.runner.started.wait(5)
    assert stub.Health(vlm_pb2.HealthRequest(), timeout=3).ready
    service.runner.release.set()
    assert _status(first) == "OK"


def test_a_queued_request_whose_deadline_expires_never_reaches_the_model(serve):
    service, stub = serve(2)
    first = stub.Infer.future(_request("q0"), timeout=10)
    assert service.runner.started.wait(5)
    late = stub.Infer.future(_request("q1"), timeout=0.2)
    assert _status(late) == "DEADLINE_EXCEEDED"
    time.sleep(0.3)                     # let the server-side handler notice the expiry
    service.runner.release.set()
    assert _status(first) == "OK"
    time.sleep(0.3)                     # unrepaired code would run q1 here
    assert service.runner.calls == ["q0"]


def test_a_queued_request_cancelled_by_its_client_never_reaches_the_model(serve):
    service, stub = serve(2)
    first = stub.Infer.future(_request("q0"), timeout=10)
    assert service.runner.started.wait(5)
    gone = stub.Infer.future(_request("q1"), timeout=10)
    time.sleep(0.1)
    assert gone.cancel()
    time.sleep(0.3)
    service.runner.release.set()
    assert _status(first) == "OK"
    time.sleep(0.3)
    assert service.runner.calls == ["q0"]
