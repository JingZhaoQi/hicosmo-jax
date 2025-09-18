import os
import subprocess
import sys


def _run_python(code: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_config_init_sets_requested_cpu_cores():
    system_cores = os.cpu_count() or 1
    target_cores = min(system_cores, 4) if system_cores >= 2 else 1
    env = os.environ.copy()
    env.pop("XLA_FLAGS", None)
    env["HICOSMO_DISABLE_AUTO_XLA"] = "1"
    env["HICOSMO_TEST_TARGET_CORES"] = str(target_cores)

    script = """
import os
from hicosmo.samplers.init import Config
Config.reset()
Config.init(cpu_cores=int(os.environ['HICOSMO_TEST_TARGET_CORES']), verbose=False)
import jax
print(jax.local_device_count())
"""

    result = _run_python(script, env)
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert output_lines, f"No output captured. stderr: {result.stderr}"
    reported = int(output_lines[-1])
    assert reported == target_cores, (
        f"Expected {target_cores} JAX devices, got {reported}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
