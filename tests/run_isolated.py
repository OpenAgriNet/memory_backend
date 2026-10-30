"""Run chunk contracts in a disposable Qdrant container; never uses the demo DB."""
import os
import socket
import tempfile
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

service = None
name = "amul-chunk-tests-" + uuid.uuid4().hex[:8]
try:
    subprocess.run(["docker", "run", "--rm", "-d", "--name", name,
                    "-p", "127.0.0.1::6333", "qdrant/qdrant:latest"], check=True, stdout=subprocess.DEVNULL)
    address = subprocess.check_output(["docker", "port", name, "6333/tcp"], text=True).strip()
    endpoint = "http://" + address
    for attempt in range(60):
        try:
            urllib.request.urlopen(endpoint + "/readyz", timeout=1).close()
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError("test Qdrant did not become ready")
    env = dict(os.environ, AMUL_CHUNK_TEST_QDRANT=endpoint,
               AMUL_TASK_STATE_TEST_QDRANT=endpoint, QDRANT_URL=endpoint, QDRANT_API_KEY="")
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_episode_chunks.py", "-v"],
                            cwd=Path(__file__).resolve().parents[1], env=env)
    if result.returncode == 0:
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_task_processor.py", "-v"],
                                cwd=Path(__file__).resolve().parents[1], env=env)
    if result.returncode == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env.update(QDRANT_URL=endpoint, QDRANT_HTTP=endpoint, QDRANT_COLLECTION="isolated_contracts",
                   MEMORY_API_URL=f"http://127.0.0.1:{port}", AMUL_MEMORY_INTEGRATION="1")
        with tempfile.TemporaryFile() as log:
            service = subprocess.Popen([sys.executable, "-m", "uvicorn", "fixture_service:app", "--host", "127.0.0.1",
                                        "--port", str(port)], cwd=Path(__file__).resolve().parent,
                                       env=env, stdout=log, stderr=log)
            for attempt in range(100):
                try:
                    urllib.request.urlopen(env["MEMORY_API_URL"] + "/health", timeout=1).close()
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                log.seek(0)
                raise RuntimeError(log.read().decode())
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_service.py", "-v"],
                                    cwd=Path(__file__).resolve().parents[1], env=env)
finally:
    if service is not None:
        service.terminate()
        service.wait(timeout=10)
    subprocess.run(["docker", "stop", name], stdout=subprocess.DEVNULL, check=False)
sys.exit(result.returncode)
