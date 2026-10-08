"""Resource residency invalidation without weights or inference."""
from functools import lru_cache
import mlx.core as mx
import pytest

import gsovits_mlx.io as io
import gsovits_mlx.text.bert as bert
import gsovits_mlx.text.bert_tokenizer as tokenizer
from gsovits_mlx.text.preproc import TextFrontend, _BERT_RESOURCES


@pytest.fixture
def frontend_export(tmp_path, monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    _BERT_RESOURCES.clear()
    root = tmp_path / "models"
    directory = root / "bert"
    directory.mkdir(parents=True)
    for name, text in (("bert.safetensors", "export"), ("config.json", "{}"),
                       ("tokenizer.json", "tokenizer")):
        (directory / name).write_text(text)
    calls = []
    monkeypatch.setattr(io, "load_mlx_safetensors", lambda path: calls.append(path) or {})
    monkeypatch.setattr(bert, "BertModel", lambda arrays, cfg: object())

    @lru_cache(maxsize=4)
    def load_tokenizer(path):
        return object()
    monkeypatch.setattr(tokenizer, "load_bert_tokenizer", load_tokenizer)
    monkeypatch.setenv("GSOVITS_FRONTEND_MODEL_CACHE", "1")
    monkeypatch.setenv("GSOVITS_FRONTEND_F32", "0")
    yield root, directory, calls
    _BERT_RESOURCES.clear()
    mx.set_default_device(previous)


def frontend(root):
    fe = TextFrontend.__new__(TextFrontend)
    fe.models_root, fe._bert, fe._tok = str(root), None, None
    return fe


def test_model_reused_between_frontends_and_reloaded_after_export_change(frontend_export):
    root, directory, calls = frontend_export
    a, b = frontend(root), frontend(root)
    assert a._load_bert() is b._load_bert()
    assert a._tok is b._tok
    assert len(calls) == 1
    (directory / "bert.safetensors").write_text("replacement export")
    c = frontend(root)
    assert c._load_bert() is not a._bert
    assert c._tok is not a._tok
    assert len(calls) == 2 and len(_BERT_RESOURCES) == 1


def test_precision_mode_and_disabled_cache(frontend_export, monkeypatch):
    root, _, calls = frontend_export
    a = frontend(root)
    a._load_bert()
    monkeypatch.setenv("GSOVITS_FRONTEND_F32", "1")
    b = frontend(root)
    assert b._load_bert() is not a._bert
    monkeypatch.setenv("GSOVITS_FRONTEND_MODEL_CACHE", "0")
    c, d = frontend(root), frontend(root)
    assert c._load_bert() is not d._load_bert()
    assert len(calls) == 4
    assert not _BERT_RESOURCES
