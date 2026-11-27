import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = json.load(open(os.path.join(os.path.dirname(__file__), "data", "reference_synthetic.json")))


def cli(*args, check=True):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", "vmim21cm", *map(str, args)], check=check, cwd=ROOT,
                          env=env, capture_output=True, text=True)


def test_cli_help_and_dry_run(synth):
    assert "stage1" in cli("--help").stdout
    out = cli("run", synth[1]["patha"], "--dry-run", "--set", "nle.n_layers=3").stdout
    assert "stage1" in out and "stage2" in out and "stage3" in out and "nle.n_layers=3" in out


@pytest.mark.slow
def test_full_pipeline_reproduces_reference(tmp_path):
    """Stages 1-4 on synthetic data (CPU); losses must match the original research code."""
    d = str(tmp_path)
    cli("demo", d)
    p = {n: f"{d}/{n}.yaml" for n in ("baseline", "patha", "pathb")}
    cli("stage1", p["patha"], "--cseed", 43, "--device", "cpu")
    cli("stage1", p["pathb"], "--cseed", 43, "--train-only", "--device", "cpu")
    for k in range(2):
        cli("stage1", p["pathb"], "--cseed", 43, "--export-only", "--shard", k, "--nshard", 2, "--device", "cpu")
    cli("merge", p["pathb"], "--cseed", 43)
    for name in p:
        cs = 0 if name == "baseline" else 43
        cli("stage2", p[name], "--cseed", cs, "--nseed", 42)
        for task in range(2):
            cli("stage3", p[name], "--cseed", cs, "--nseed", 42, "--task", task, "--ntasks", 2)
    out = cli("stage4", f"{d}/eval.yaml").stdout
    assert "3 runs, 16 shared simulations" in out

    o = f"{d}/out"
    got = {"patha_compressor_loss": f"{o}/patha/c43/compressor", "pathb_compressor_loss": f"{o}/pathb/c43/compressor",
           "baseline_nle_loss": f"{o}/baseline/raw/nle_n42", "patha_nle_loss": f"{o}/patha/c43/nle_n42",
           "pathb_nle_loss": f"{o}/pathb/c43/nle_n42"}
    for key, path in got.items():
        loss = np.load(f"{path}/loss_history.npy")[:, 1:]
        # bit-identical with the paper software versions; small drift allowed across versions/CPUs
        assert np.allclose(loss, REF[key], rtol=1e-3, atol=1e-4), key
    assert os.path.exists(f"{o}/pathb/c43/compressor/cube_stats.npz")

    from vmim21cm.cnn import CubeCompressor
    t = np.load(f"{o}/pathb/c43/summaries/t.npy")
    assert t.shape == (48 * 6, 4) and np.all(np.isfinite(t))
    assert CubeCompressor.from_run(f"{o}/pathb/c43").t_dim == 4


@pytest.mark.slow
def test_local_runner(tmp_path):
    d = str(tmp_path)
    cli("demo", d)
    r = cli("run", f"{d}/patha.yaml", "--device", "cpu", "--eval")
    assert os.path.exists(f"{d}/out/eval/patha/metrics.csv"), r.stdout[-2000:]
