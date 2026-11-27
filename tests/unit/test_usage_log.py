import json

import httpx

from ra_agent.config import ModelSettings
from ra_agent.llm import client as client_mod


def test_post_logs_usage(tmp_path, monkeypatch):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setattr(client_mod, "USAGE_LOG", log)
    reply = {"choices": [{"message": {"content": "hi"}}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(200, json=reply, request=httpx.Request("POST", "http://x")))
    c = client_mod.OpenAICompatibleVisionClient(ModelSettings(base_url="http://x", model="m", api_key_env="K"), "key")
    assert c.complete("p") == "hi"
    line = json.loads(log.read_text())
    assert line["model"] == "m" and line["total_tokens"] == 10
    assert c.usage["calls"] == 1 and c.usage["total_tokens"] == 10
