"""Bản clone mới không có mã sinh từ .proto. Test này đi đúng đường đó trong CI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_stub_duoc_sinh_lai_va_import_duoc():
    from serve.gen_proto import ensure_stubs, OUTPUTS
    ensure_stubs()
    assert all(p.exists() for p in OUTPUTS)
    from serve import vlm_pb2, vlm_pb2_grpc   # import tương đối phải chạy được
    req = vlm_pb2.InferRequest(question="q", max_edge=768)
    assert req.max_edge == 768
    assert hasattr(vlm_pb2_grpc, "VlmServiceStub")


def test_stub_dung_import_tuong_doi():
    from serve.gen_proto import ensure_stubs, OUTPUTS
    ensure_stubs()
    assert "from . import vlm_pb2" in OUTPUTS[1].read_text()
