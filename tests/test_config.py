import pytest
import yaml

from vmim21cm.config import apply_overrides, load_config, resolve_device, run_dir, summaries_path


def _write(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("VMIM_TEST_ROOT", "/data/x")
    cfg = load_config(_write(tmp_path, {"out_root": "${VMIM_TEST_ROOT}/out", "n": [1, "${VMIM_TEST_ROOT}"]}))
    assert cfg["out_root"] == "/data/x/out" and cfg["n"] == [1, "/data/x"]


def test_unset_env_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("VMIM_TEST_UNSET", raising=False)
    with pytest.raises(KeyError):
        load_config(_write(tmp_path, {"a": "${VMIM_TEST_UNSET}/b"}))


def test_override_replaces_unset_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("VMIM_TEST_UNSET", raising=False)
    cfg = load_config(_write(tmp_path, {"data": {"p": "${VMIM_TEST_UNSET}"}}), ["data.p=/here"])
    assert cfg["data"]["p"] == "/here"


def test_overrides_parse_yaml_and_create_keys():
    cfg = apply_overrides({"nle": {"n_layers": 4}}, ["nle.n_layers=8", "new.key=[1, 2]", "x=1.0e-2"])
    assert cfg["nle"]["n_layers"] == 8 and cfg["new"]["key"] == [1, 2] and cfg["x"] == 0.01
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_run_dirs():
    cfg = {"out_root": "/o", "name": "patha/vmim_k4", "compressor": {"type": "mlp"}, "data": {}}
    assert str(run_dir(cfg, 43)) == "/o/patha/vmim_k4/c43"
    assert str(summaries_path(cfg, 43)) == "/o/patha/vmim_k4/c43/summaries"
    base = {"out_root": "/o", "name": "b", "compressor": {"type": "none"}, "data": {"summaries_npz": "/s.npz"}}
    assert str(run_dir(base, 0)) == "/o/b/raw" and str(summaries_path(base, 0)) == "/s.npz"


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
