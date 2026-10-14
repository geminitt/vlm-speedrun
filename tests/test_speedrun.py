"""speedrun.sh orchestration, with pixi and docker replaced by recording stubs.

The script runs in a temporary folder, so its relative output paths never touch the
repository's results/. The stubs print nothing real: these tests check which
commands the script issues and in what order, not what those commands produce.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "speedrun.sh"
SCRIPT_SETTINGS = ("FAST", "DOCKER", "START", "END", "MODEL", "PIXI", "FAIL_DOCKER_CLIENT")

# pixi stub: answers the script's two probes (the interpreter path is a stand-in
# server that waits to be killed), records every other command, and fails the
# Docker client benchmark when FAIL_DOCKER_CLIENT=1.
PIXI = r"""#!/bin/bash
if [[ "$3" == "-c" ]]; then
  echo "probe $4" >> "$LOG"
  if [[ "$4" == *sys.executable* ]]; then echo "$(dirname "$0")/server"; else echo 1; fi
  exit 0
fi
echo "pixi $*" >> "$LOG"
if [[ "${FAIL_DOCKER_CLIENT:-0}" == "1" && "$*" == *gate4_docker_* ]]; then exit 3; fi
# Like the real client, which waits for Health: return only once the native server
# it talks to has started, or the script may kill that server before it logs.
if [[ "$*" == *gate4_serving_* ]]; then
  want=$(grep -c "gate4_serving_" "$LOG")
  for _ in $(seq 100); do
    [[ $(grep -c "^server " "$LOG") -ge $want ]] && break
    sleep 0.05
  done
fi
"""
SERVER = r"""#!/bin/bash
echo "server $*" >> "$LOG"
exec sleep 30
"""
DOCKER = r"""#!/bin/bash
echo "docker $*" >> "$LOG"
if [[ "$1" == "run" ]]; then echo fake-container-id; fi
"""


@pytest.fixture
def run(tmp_path):
    if not shutil.which("bash"):
        pytest.skip("bash is required")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("pixi", PIXI), ("docker", DOCKER), ("server", SERVER)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "commands.log"

    def go(**env):
        # Step 0 of speedrun.sh runs this suite: its own FAST, DOCKER, range and model
        # settings must not leak into the script under test.
        inherited = {k: v for k, v in os.environ.items() if k not in SCRIPT_SETTINGS}
        full = dict(inherited, PIXI=str(bin_dir / "pixi"), LOG=str(log),
                    PATH=f"{bin_dir}:{os.environ['PATH']}", **env)
        proc = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=full,
                              capture_output=True, text=True, timeout=30)
        return proc, (log.read_text().splitlines() if log.exists() else [])
    return go


def test_end_excludes_the_report_step(run):
    proc, cmds = run(START="0", END="0")
    assert proc.returncode == 0, proc.stderr
    assert any("pytest" in c for c in cmds)
    assert not any("bench.plot" in c or "bench.report" in c for c in cmds)


def test_the_report_step_runs_when_it_is_in_range(run):
    proc, cmds = run(START="9", END="9")
    assert proc.returncode == 0, proc.stderr
    assert any("bench.plot" in c for c in cmds)
    assert any("bench.report --write" in c for c in cmds)
    assert not any("pytest" in c for c in cmds)


def test_docker_server_runs_the_selected_model(run):
    proc, cmds = run(START="8", END="8", DOCKER="1", MODEL="org/some-model")
    assert proc.returncode == 0, proc.stderr
    runs = [c for c in cmds if c.startswith("docker run")]
    assert len(runs) == 2
    assert all("--model org/some-model" in c for c in runs)


def test_a_failing_docker_client_still_removes_its_container(run):
    proc, cmds = run(START="8", END="8", DOCKER="1", FAIL_DOCKER_CLIENT="1")
    assert proc.returncode != 0                      # the failure still stops the script
    assert "docker rm -f fake-container-id" in cmds


@pytest.mark.parametrize("start, end, needed", [("0", "0", False), ("8", "9", False),
                                                ("3", "3", True), ("7", "7", True)])
def test_the_noise_level_is_read_only_by_steps_that_use_it(run, start, end, needed):
    # Steps 3, 4, 6 and 7 record the gate-0 noise; reading it in any other range would
    # require a gate-0 file that the requested steps never produce.
    proc, cmds = run(START=start, END=end)
    assert proc.returncode == 0, proc.stderr
    assert any("robust_cv" in c for c in cmds) == needed


def test_benchmark_servers_listen_on_localhost_only(run):
    # The client runs on the same machine; a benchmark has no reason to expose an
    # unauthenticated server to the network.
    proc, cmds = run(START="8", END="8", DOCKER="1")
    assert proc.returncode == 0, proc.stderr
    docker_runs = [c for c in cmds if c.startswith("docker run")]
    assert docker_runs and all("-p 127.0.0.1:50051:50051" in c for c in docker_runs)
    natives = [c for c in cmds if c.startswith("server ")]
    assert len(natives) == 2 and all("--host 127.0.0.1" in c for c in natives)
    clients = [c for c in cmds if "serve.client_bench" in c]
    assert len(clients) == 4 and all("--target 127.0.0.1:50051" in c for c in clients)
