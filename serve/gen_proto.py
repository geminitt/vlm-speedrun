"""Sinh mã Python từ vlm.proto.

Hai file vlm_pb2.py và vlm_pb2_grpc.py không nằm trong git vì chúng là mã sinh tự
động. File này đảm bảo chúng luôn được tạo lại khi thiếu, để một bản clone mới vẫn
chạy được máy chủ.

    python -m serve.gen_proto        # sinh lại một cách tường minh
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTO = HERE / "vlm.proto"
OUTPUTS = (HERE / "vlm_pb2.py", HERE / "vlm_pb2_grpc.py")


def generate():
    from grpc_tools import protoc
    code = protoc.main([
        "protoc", f"-I{HERE}",
        f"--python_out={HERE}", f"--grpc_python_out={HERE}", str(PROTO),
    ])
    if code != 0:
        raise RuntimeError(f"protoc thất bại với mã {code}")
    # protoc sinh ra import tuyệt đối "import vlm_pb2", chỉ chạy được khi thư mục
    # serve/ nằm trong sys.path. Đổi thành import tương đối trong package.
    grpc_file = OUTPUTS[1]
    src = grpc_file.read_text()
    grpc_file.write_text(src.replace("import vlm_pb2 as vlm__pb2",
                                     "from . import vlm_pb2 as vlm__pb2"))


def _version(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _generated_grpc_version():
    """Phiên bản grpcio mà mã sinh ra đòi hỏi, ghi sẵn trong file do protoc tạo."""
    m = re.search(r"GRPC_GENERATED_VERSION = '([\d.]+)'", OUTPUTS[1].read_text())
    return m.group(1) if m else None


def ensure_stubs():
    """Sinh lại stub khi thiếu, cũ hơn file .proto, hoặc lệch phiên bản grpcio.

    Trường hợp lệch phiên bản xảy ra khi máy có hai môi trường: stub sinh bởi
    grpcio-tools bản mới hơn sẽ từ chối chạy với grpcio bản cũ hơn.
    """
    src_time = PROTO.stat().st_mtime
    if all(p.exists() and p.stat().st_mtime >= src_time for p in OUTPUTS):
        import grpc
        needed = _generated_grpc_version()
        if needed is None or _version(needed) <= _version(grpc.__version__):
            return
    generate()


if __name__ == "__main__":
    generate()
    print("đã sinh:", ", ".join(p.name for p in OUTPUTS))
