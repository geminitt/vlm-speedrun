"""A fresh clone has no code generated from .proto. This test walks that path in CI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_stubs_are_regenerated_and_importable():
    from serve.gen_proto import ensure_stubs, OUTPUTS
    ensure_stubs()
    assert all(p.exists() for p in OUTPUTS)
    from serve import vlm_pb2, vlm_pb2_grpc   # the relative import must work
    req = vlm_pb2.InferRequest(question="q", max_edge=768)
    assert req.max_edge == 768
    assert hasattr(vlm_pb2_grpc, "VlmServiceStub")


def test_stubs_use_a_relative_import():
    from serve.gen_proto import ensure_stubs, OUTPUTS
    ensure_stubs()
    assert "from . import vlm_pb2" in OUTPUTS[1].read_text()
