"""Model-resource cache contracts; no weights, arrays or GPU execution."""
import mlx.core as mx
import pytest

from gsovits_mlx.model_cache import clear_model_cache, resident_model


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    clear_model_cache()
    monkeypatch.setenv("GSOVITS_MODEL_CACHE", "1")
    monkeypatch.setenv("GSOVITS_AR_MIXED_GEMV", "1")
    monkeypatch.setattr(mx, "default_device", lambda: "cpu")
    yield
    clear_model_cache()


def export(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "weights.safetensors").write_text("export")
    (root / "config.json").write_text("{}")
    return root


def test_residency_export_reload_and_explicit_release(tmp_path):
    root = export(tmp_path)
    calls = []
    @resident_model("gpt")
    def load(path):
        calls.append(path)
        return object()
    a = load(root)
    assert load(root) is a and len(calls) == 1
    (root / "weights.safetensors").write_text("replacement export")
    b = load(root)
    assert b is not a and len(calls) == 2
    clear_model_cache()
    assert load(root) is not b and len(calls) == 3


def test_category_shared_between_loader_families_and_modes(tmp_path, monkeypatch):
    root = export(tmp_path)
    @resident_model("sovits")
    def v2(path, dtype=None):
        return object()
    @resident_model("sovits")
    def v3(path, dtype=None):
        return object()
    a = v2(root)
    assert v2(root) is a
    b = v3(root)
    assert v3(root) is b and v2(root) is not a
    half = v2(root, dtype=mx.float16)
    assert v2(root, dtype=mx.float16) is half
    full = v2(root, dtype=mx.float32)
    assert full is not half
    monkeypatch.setattr(mx, "default_device", lambda: "gpu")
    assert v2(root, dtype=mx.float32) is not full


def test_constructor_flags_and_cache_disable(tmp_path, monkeypatch):
    root = export(tmp_path)
    @resident_model("gpt")
    def load(path):
        return object()
    a = load(root)
    monkeypatch.setenv("GSOVITS_AR_MIXED_GEMV", "0")
    b = load(root)
    assert b is not a
    monkeypatch.setenv("GSOVITS_MODEL_CACHE", "0")
    assert load(root) is not load(root)
    monkeypatch.setenv("GSOVITS_MODEL_CACHE", "1")
    assert load(root) is not b


def test_loader_can_fingerprint_nested_exports(tmp_path):
    import json
    root = export(tmp_path)
    nested = root / "metadata"
    nested.mkdir()
    config = nested / "config.json"
    config.write_text('{"value": 1}')
    calls = []
    @resident_model("custom", patterns=("**/*.json", "*.safetensors"))
    def load(path):
        calls.append(path)
        return json.loads((nested / "config.json").read_text())
    assert load(root)["value"] == 1
    assert load(root)["value"] == 1 and len(calls) == 1
    config.write_text('{"value": 20}')
    assert load(root)["value"] == 20 and len(calls) == 2

