"""Every script that loads the model or a dataset does so at a resolved commit.

The loaders are replaced by recorders that stop the script right after loading, so
these tests check which commit each script asks for, not what it measures.
"""
import copy
import importlib
import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REV_MODEL, REV_DATA = "m" * 40, "d" * 40


class Stop(Exception):
    """Raised by a fake loader: the part under test is done."""


@pytest.fixture
def pinned(monkeypatch):
    import bench.harness as h
    monkeypatch.setattr(h, "resolve_revision",
                        lambda repo, rev="main", kind="model": REV_MODEL if kind == "model" else REV_DATA)
    monkeypatch.setattr(h, "source_digest", lambda *a, **k: "src")
    monkeypatch.setattr(h, "environment_identity", lambda: {"torch": "1"})


def _fake_loaders(monkeypatch, mod, loads, stop_at_data=True):
    class FakeRunner:
        def __init__(self, model_id, **kw):
            loads.append(("model", kw.get("revision")))
            self.model = SimpleNamespace(model=None)

        def infer(self, sample, cfg):
            return "x", 1, 1

    def fake_samples(n, seed=0, dataset="chartqa", revision=None):
        loads.append(("dataset", revision))
        if stop_at_data:
            raise Stop
        return [{"sample_id": i, "query": "q", "image": None, "gold": "x", "golds": ["x"]}
                for i in range(n)]

    monkeypatch.setattr(mod, "Runner", FakeRunner, raising=False)
    monkeypatch.setattr(mod, "load_samples", fake_samples)
    monkeypatch.setattr(mod, "check_gpu_idle", lambda: None, raising=False)


@pytest.mark.parametrize("module", ["bench.breakdown", "bench.preprocess_cost",
                                    "bench.quantize", "bench.sanity_docvqa"])
def test_measurement_scripts_load_model_and_data_at_the_resolved_commit(
        module, pinned, monkeypatch, tmp_path):
    mod = importlib.import_module(module)
    loads = []
    _fake_loaders(monkeypatch, mod, loads)
    monkeypatch.setattr("sys.argv", ["x", "--out", str(tmp_path / "out.json")])
    with pytest.raises(Stop):
        mod.main()
    assert loads == [("model", REV_MODEL), ("dataset", REV_DATA)]


def test_a_script_records_its_identity_in_the_output(pinned, monkeypatch, tmp_path):
    import bench.sanity_docvqa as mod
    _fake_loaders(monkeypatch, mod, [], stop_at_data=False)
    out = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv", ["x", "--samples", "3", "--out", str(out)])
    mod.main()
    ident = json.loads(out.read_text())["identity"]
    assert ident == {"model_revision": REV_MODEL, "dataset_revision": REV_DATA,
                     "source_sha256": "src", "environment": {"torch": "1"}}


def test_latency_probe_loads_the_resolved_commit(pinned, monkeypatch, tmp_path):
    import bench.latency_probe as mod
    loads = []

    class Loader:
        def __init__(self, kind):
            self.kind = kind

        def from_pretrained(self, model_id, **kw):
            loads.append((self.kind, kw.get("revision")))
            if self.kind == "model":
                raise Stop
            return object()

    fake = types.ModuleType("transformers")
    fake.AutoProcessor, fake.AutoModelForImageTextToText = Loader("processor"), Loader("model")
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(mod, "check_gpu_idle", lambda: None)
    monkeypatch.setattr("sys.argv", ["x", "--out", str(tmp_path / "out.json")])
    with pytest.raises(Stop):
        mod.main()
    assert loads == [("processor", REV_MODEL), ("model", REV_MODEL)]


def test_client_bench_loads_the_resolved_dataset_commit(pinned, monkeypatch, tmp_path):
    import serve.client_bench as mod
    loads = []
    _fake_loaders(monkeypatch, mod, loads)

    class FakeStub:
        def __init__(self, channel):
            pass

        def Health(self, *a, **k):
            return SimpleNamespace(model=f"org/m@{REV_MODEL}", device="cuda")

    monkeypatch.setattr(mod.vlm_pb2_grpc, "VlmServiceStub", FakeStub)
    monkeypatch.setattr("sys.argv", ["x", "--out", str(tmp_path / "out.json")])
    with pytest.raises(Stop):
        mod.main()
    assert loads == [("dataset", REV_DATA)]


@pytest.mark.parametrize("recorded, expected", [(REV_DATA[:-1] + "e", REV_DATA[:-1] + "e"),
                                                (None, REV_DATA)])
def test_rescore_reads_the_dataset_commit_the_run_used(pinned, monkeypatch, tmp_path,
                                                       recorded, expected):
    import bench.rescore as mod
    seen = []

    def load_dataset(path, split=None, revision=None, **kw):
        seen.append(revision)
        raise Stop

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    run = {"dataset": "chartqa", "configs": {}, "records": []}
    if recorded:
        run["identity"] = {"dataset_revision": recorded}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    monkeypatch.setattr("sys.argv", ["x", str(path)])
    with pytest.raises(Stop):
        mod.main()
    assert seen == [expected]


def test_server_loads_and_reports_the_resolved_model_commit(pinned, monkeypatch):
    import argparse
    import bench.harness as h
    from serve import server
    seen = {}

    class FakeRunner:
        def __init__(self, model_id, **kw):
            seen.update(kw)

    monkeypatch.setattr(h, "Runner", FakeRunner)

    def fake_build(service, max_queue):
        seen["service"] = service
        raise Stop

    monkeypatch.setattr(server, "build_server", fake_build)
    args = argparse.Namespace(model="org/m", model_revision="main", max_edge=1536,
                              max_new_tokens=32, device="cpu", prompt="vi", max_queue=2, port=0)
    with pytest.raises(Stop):
        server.serve(args)
    assert seen["revision"] == REV_MODEL
    assert seen["service"].Health(None, None).model == f"org/m@{REV_MODEL}"


def test_report_ignores_identity_records(tmp_path):
    from bench.report import values
    results = tmp_path / "results"
    shutil.copytree(ROOT / "results", results)
    reference = values(ROOT / "results")
    for f in results.glob("*.json"):
        data = json.loads(f.read_text())
        data["identity"] = {"model_revision": REV_MODEL, "environment": {"torch": "1"}}
        f.write_text(json.dumps(data))
    assert values(results) == reference
