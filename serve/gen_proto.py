"""Generate Python code from vlm.proto.

vlm_pb2.py and vlm_pb2_grpc.py are not tracked in git because they are generated.
This module makes sure they are recreated whenever they are missing, so a fresh
clone can still run the server.

    python -m serve.gen_proto        # regenerate explicitly
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
        raise RuntimeError(f"protoc failed with exit code {code}")
    # protoc emits an absolute "import vlm_pb2", which only works when serve/ is on
    # sys.path. Rewrite it as a relative import inside the package.
    grpc_file = OUTPUTS[1]
    src = grpc_file.read_text()
    grpc_file.write_text(src.replace("import vlm_pb2 as vlm__pb2",
                                     "from . import vlm_pb2 as vlm__pb2"))


def _version(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _generated_grpc_version():
    """The grpcio version the generated code requires, as recorded by protoc."""
    m = re.search(r"GRPC_GENERATED_VERSION = '([\d.]+)'", OUTPUTS[1].read_text())
    return m.group(1) if m else None


def ensure_stubs():
    """Regenerate the stubs when missing, older than the .proto, or grpcio-mismatched.

    A version mismatch happens on machines with two environments: stubs generated
    by a newer grpcio-tools refuse to load under an older grpcio.
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
    print("generated:", ", ".join(p.name for p in OUTPUTS))
