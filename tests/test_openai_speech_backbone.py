"""apps.openai_speech Engine: VIENEU_BACKBONE_REPO wiring and its ONNX guard.

``Vieneu`` is stubbed at the module level (``apps.openai_speech`` imports it with
``from vieneu import Vieneu``), so this never loads the real model or torch.
"""
import pytest

from apps import openai_speech as sp


class _FakeScheduler:
    B = 4


class _FakeTts:
    """Just enough of V3TurboVieNeuTTS for Engine.__init__: backend, the warm-up
    stream, and the PyTorch stream scheduler."""

    def __init__(self, backend):
        self.backend = backend
        self._preset_voices = {}
        self.text_rules = True
        self._pronunciation_overrides = None

    def _get_stream_scheduler(self):
        return _FakeScheduler()

    def infer_stream(self, text, apply_watermark=False):
        return iter(())


def _fake_vieneu_factory(backend):
    """A stand-in for ``vieneu.Vieneu`` that records the kwargs it was called
    with, on the returned callable itself (``.kwargs``)."""

    def _factory(mode="v3turbo", **kw):
        assert mode == "v3turbo"
        _factory.kwargs = kw
        return _FakeTts(backend)

    return _factory


@pytest.fixture(autouse=True)
def _no_leftover_singleton():
    # Engine() is called directly in every test below, bypassing the module's
    # get-or-create singleton (`engine()`) — but clear it anyway in case a test
    # elsewhere in the suite process left it set.
    sp.ENGINE = None
    yield
    sp.ENGINE = None


def test_backbone_repo_unset_by_default(monkeypatch):
    """No VIENEU_BACKBONE_REPO set: `backbone_repo` is left out of the call, so
    V3TurboVieNeuTTS keeps its own default."""
    monkeypatch.delenv("VIENEU_BACKBONE_REPO", raising=False)
    fake = _fake_vieneu_factory("pytorch")
    monkeypatch.setattr(sp, "Vieneu", fake)

    sp.Engine()

    assert "backbone_repo" not in fake.kwargs


def test_backbone_repo_passed_through_on_pytorch(monkeypatch):
    monkeypatch.setenv("VIENEU_BACKBONE_REPO", "C:\\models\\minh-quan-pro-tuned")
    fake = _fake_vieneu_factory("pytorch")
    monkeypatch.setattr(sp, "Vieneu", fake)

    engine = sp.Engine()

    assert fake.kwargs["backbone_repo"] == "C:\\models\\minh-quan-pro-tuned"
    assert engine.backend == "pytorch"


def test_backbone_repo_on_onnx_backend_fails_at_startup(monkeypatch):
    """A custom backbone has no ONNX export; resolving to the ONNX backend
    anyway (no CUDA torch, or VIENEU_BACKEND/VIENEU_DEVICE forced CPU) must
    fail loudly instead of silently serving the wrong (default) voice."""
    monkeypatch.setenv("VIENEU_BACKBONE_REPO", "C:\\models\\minh-quan-pro-tuned")
    fake = _fake_vieneu_factory("onnx")
    monkeypatch.setattr(sp, "Vieneu", fake)

    with pytest.raises(RuntimeError, match="VIENEU_BACKBONE_REPO"):
        sp.Engine()


def test_no_backbone_repo_on_onnx_backend_is_unaffected(monkeypatch):
    """The guard only fires when VIENEU_BACKBONE_REPO is actually set: plain
    CPU/ONNX serving (today's default) must keep working."""
    monkeypatch.delenv("VIENEU_BACKBONE_REPO", raising=False)
    fake = _fake_vieneu_factory("onnx")
    monkeypatch.setattr(sp, "Vieneu", fake)

    engine = sp.Engine()

    assert engine.backend == "onnx"


def test_text_rules_on_by_default(monkeypatch):
    monkeypatch.delenv("VIENEU_TEXT_RULES", raising=False)
    fake = _fake_vieneu_factory("onnx")
    monkeypatch.setattr(sp, "Vieneu", fake)

    sp.Engine()

    assert fake.kwargs["text_rules"] is True


def test_text_rules_disabled_by_env(monkeypatch):
    monkeypatch.setenv("VIENEU_TEXT_RULES", "0")
    fake = _fake_vieneu_factory("onnx")
    monkeypatch.setattr(sp, "Vieneu", fake)

    sp.Engine()

    assert fake.kwargs["text_rules"] is False


def test_pronunciations_path_passed_through(monkeypatch):
    monkeypatch.setenv("VIENEU_PRONUNCIATIONS", "C:\\config\\pronunciations.txt")
    fake = _fake_vieneu_factory("onnx")
    monkeypatch.setattr(sp, "Vieneu", fake)

    sp.Engine()

    assert fake.kwargs["pronunciations"] == "C:\\config\\pronunciations.txt"


def test_pronunciations_unset_by_default(monkeypatch):
    monkeypatch.delenv("VIENEU_PRONUNCIATIONS", raising=False)
    fake = _fake_vieneu_factory("onnx")
    monkeypatch.setattr(sp, "Vieneu", fake)

    sp.Engine()

    assert "pronunciations" not in fake.kwargs
