"""The speech endpoint's stream slot is released however the response ends.

With one slot (the CPU default), a single leaked slot makes every later request
429 until restart — e.g. a client that times out while its request is queued,
so the response is sent to a closed connection and the audio generator never
starts.
"""
import asyncio
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps import openai_speech as api


class _FakeTTS:
    def resolve_voice_name(self, name):
        return name

    def infer_stream(self, text, **kwargs):
        yield np.zeros(2400, np.float32)


def _fake_engine(max_streams=1):
    eng = object.__new__(api.Engine)
    eng.tts = _FakeTTS()
    eng.backend = "fake"
    eng.watermark = False
    eng.max_streams = max_streams
    eng.max_queue = max_streams
    eng.queue_timeout = 0.05
    eng._gate = threading.BoundedSemaphore(max_streams)
    eng._lock = threading.Lock()
    eng.active = 0
    eng.waiting = 0
    return eng


@pytest.fixture
def eng(monkeypatch):
    fake = _fake_engine()
    monkeypatch.setattr(api, "ENGINE", fake)
    monkeypatch.delenv("VIENEU_API_KEY", raising=False)
    return fake


def test_slot_released_after_a_completed_stream(eng):
    client = TestClient(api.app)
    for _ in range(3):  # a leaked slot would 429 the second request
        r = client.post("/v1/audio/speech", json={"input": "Xin chào.", "response_format": "pcm"})
        assert r.status_code == 200
        assert len(r.content) > 0
    assert eng.active == 0


def test_slot_released_when_client_left_before_the_body_started(eng):
    started = []

    def body():
        started.append(True)
        yield b"x"

    eng.acquire()
    slot = api._Slot(eng)
    response = api._SlotStreamingResponse(body(), slot, "spk-test", media_type="audio/pcm")

    async def closed_connection_send(message):
        raise OSError("client disconnected")

    async def receive():
        return {"type": "http.disconnect"}

    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
    with pytest.raises(Exception):
        asyncio.run(response(scope, receive, closed_connection_send))
    assert not started
    assert eng.active == 0
    eng.acquire()  # the slot is free again, no 429
    eng.release()


def test_slot_release_is_idempotent(eng):
    eng.acquire()
    slot = api._Slot(eng)
    assert slot.release() is True
    assert slot.release() is False
    assert eng.active == 0
