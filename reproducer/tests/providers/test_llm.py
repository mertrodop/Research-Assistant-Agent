"""Tests for reproducer.providers.llm (offline by default; live tests marked llm_live)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from reproducer.providers import llm as llm_mod
from reproducer.providers.llm import (
    LLM,
    Budget,
    BudgetExceeded,
    FakeLLM,
    LLMError,
    LLMResponse,
    MeteredLLM,
    OllamaLLM,
    OpenAICompatLLM,
    llm_for_node,
    make_llm,
    parse_spec,
)

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
SECRET = "sk-test-SECRET-123"


# --------------------------------------------------------------------------- stub server


class Stub:
    """Scripted HTTP server. routes[(method, path)] -> list of (status, body, delay_s)."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[tuple[int, Any, float]]] = {}
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                with stub.lock:
                    stub.requests.append(
                        {
                            "method": method,
                            "path": self.path,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "json": json.loads(raw) if raw else None,
                        }
                    )
                    queue = stub.routes.get((method, self.path), [])
                    status, body, delay = queue.pop(0) if queue else (404, {"error": "nf"}, 0)
                if delay:
                    time.sleep(delay)
                out = body if isinstance(body, bytes) else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def add(self, method: str, path: str, status: int, body: Any, delay: float = 0.0) -> None:
        self.routes.setdefault((method, path), []).append((status, body, delay))

    def posts(self, path: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "POST" and r["path"] == path]


@pytest.fixture
def stub() -> Iterator[Stub]:
    s = Stub()
    yield s
    s.server.shutdown()
    s.server.server_close()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Backoff sleeps are recorded instead of slept."""
    slept: list[float] = []
    monkeypatch.setattr(llm_mod, "_backoff_sleep", slept.append)
    return slept


def _ollama_ok(text: str = '{"ok": true}') -> dict[str, Any]:
    return {
        "model": "m",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 5,
    }


def _openai_ok(text: str = '{"ok": true}') -> dict[str, Any]:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
    }


# --------------------------------------------------------------------------- Ollama


def test_ollama_payload_and_parsing(stub: Stub) -> None:
    stub.add("GET", "/api/version", 200, {"version": "0.18.0"})
    stub.add("POST", "/api/chat", 200, _ollama_ok())
    client = OllamaLLM(stub.url, "qwen2.5-coder:7b", num_ctx=4096, keep_alive="10m")
    resp = client.complete("sys", "hello", max_tokens=64, json_schema=SCHEMA)

    assert isinstance(client, LLM)
    assert resp == LLMResponse(
        text='{"ok": true}',
        input_tokens=12,
        output_tokens=5,
        model="qwen2.5-coder:7b",
        backend="ollama",
        latency_s=resp.latency_s,
    )
    (req,) = stub.posts("/api/chat")
    body = req["json"]
    assert body["model"] == "qwen2.5-coder:7b"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert body["stream"] is False
    assert body["keep_alive"] == "10m"
    assert body["options"] == {"num_ctx": 4096, "num_predict": 64, "temperature": 0.0, "seed": 0}
    assert body["format"] == SCHEMA


def test_ollama_no_format_without_schema(stub: Stub) -> None:
    stub.add("POST", "/api/chat", 200, _ollama_ok("hi"))
    OllamaLLM(stub.url, "m").complete("s", "p", max_tokens=8)
    assert "format" not in stub.posts("/api/chat")[0]["json"]


@pytest.mark.parametrize("version", ["0.4.7", "garbage"])
def test_ollama_old_or_unknown_version_falls_back_to_json_mode(stub: Stub, version: str) -> None:
    stub.add("GET", "/api/version", 200, {"version": version})
    stub.add("POST", "/api/chat", 200, _ollama_ok())
    stub.add("POST", "/api/chat", 200, _ollama_ok())
    client = OllamaLLM(stub.url, "m")
    client.complete("s", "p", max_tokens=8, json_schema=SCHEMA)
    client.complete("s", "p", max_tokens=8, json_schema=SCHEMA)
    assert [r["json"]["format"] for r in stub.posts("/api/chat")] == ["json", "json"]
    # version probed once, cached
    assert sum(r["path"] == "/api/version" for r in stub.requests) == 1


def test_ollama_unload(stub: Stub) -> None:
    stub.add("POST", "/api/chat", 200, {"model": "m", "done": True, "done_reason": "unload"})
    assert OllamaLLM(stub.url, "m").unload() is True
    body = stub.posts("/api/chat")[0]["json"]
    assert body == {"model": "m", "messages": [], "keep_alive": 0, "stream": False}


def test_ollama_bad_shape_raises(stub: Stub) -> None:
    stub.add("POST", "/api/chat", 200, {"nope": 1})
    with pytest.raises(LLMError, match="response shape"):
        OllamaLLM(stub.url, "m").complete("s", "p", max_tokens=8)


# --------------------------------------------------------------------------- OpenAI-compat


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("schema", {"type": "json_schema", "json_schema": {"name": "response", "schema": SCHEMA}}),
        ("object", {"type": "json_object"}),
        ("none", None),
    ],
)
def test_openai_payload_and_response_format(stub: Stub, mode: str, expected: Any) -> None:
    stub.add("POST", "/v1/chat/completions", 200, _openai_ok())
    client = OpenAICompatLLM(f"{stub.url}/v1/", "gpt-x", json_mode=mode, env={})  # type: ignore[arg-type]
    resp = client.complete("sys", "hello", max_tokens=32, json_schema=SCHEMA)

    assert isinstance(client, LLM)
    assert (resp.text, resp.input_tokens, resp.output_tokens) == ('{"ok": true}', 20, 7)
    assert (resp.backend, resp.model) == ("openai", "gpt-x")
    (req,) = stub.posts("/v1/chat/completions")
    body = req["json"]
    assert body["model"] == "gpt-x"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["max_tokens"] == 32
    assert body["temperature"] == 0.0
    assert body.get("response_format") == expected


def test_openai_no_response_format_without_schema(stub: Stub) -> None:
    stub.add("POST", "/v1/chat/completions", 200, _openai_ok("hi"))
    OpenAICompatLLM(f"{stub.url}/v1", "g", env={}).complete("s", "p", max_tokens=4)
    assert "response_format" not in stub.posts("/v1/chat/completions")[0]["json"]


def test_openai_no_auth_header_when_key_unset(stub: Stub) -> None:
    stub.add("POST", "/v1/chat/completions", 200, _openai_ok())
    OpenAICompatLLM(f"{stub.url}/v1", "g", env={}).complete("s", "p", max_tokens=4)
    assert "authorization" not in stub.posts("/v1/chat/completions")[0]["headers"]


def test_openai_auth_header_when_key_set_and_key_never_leaks(stub: Stub) -> None:
    stub.add("POST", "/v1/chat/completions", 200, _openai_ok())
    stub.add("POST", "/v1/chat/completions", 401, {"error": "bad key"})
    env = {"MY_KEY": SECRET}
    client = OpenAICompatLLM(f"{stub.url}/v1", "g", api_key_env="MY_KEY", env=env)
    resp = client.complete("s", "p", max_tokens=4)
    assert stub.posts("/v1/chat/completions")[0]["headers"]["authorization"] == f"Bearer {SECRET}"

    with pytest.raises(LLMError) as ei:
        client.complete("s", "p", max_tokens=4)
    assert ei.value.status == 401
    assert SECRET not in str(ei.value)
    assert SECRET not in repr(client)
    assert SECRET not in resp.model_dump_json()
    assert SECRET not in json.dumps(vars(client), default=str).replace(json.dumps(env), "")


def test_openai_missing_usage_gives_zero_tokens(stub: Stub) -> None:
    stub.add("POST", "/v1/chat/completions", 200, {"choices": [{"message": {"content": "x"}}]})
    resp = OpenAICompatLLM(f"{stub.url}/v1", "g", env={}).complete("s", "p", max_tokens=4)
    assert (resp.input_tokens, resp.output_tokens) == (0, 0)


def test_openai_invalid_json_mode() -> None:
    with pytest.raises(ValueError, match="json_mode"):
        OpenAICompatLLM("http://x", "g", json_mode="xml")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- retries


def test_retry_503_then_200(stub: Stub, no_sleep: list[float]) -> None:
    stub.add("POST", "/api/chat", 503, {"error": "busy"})
    stub.add("POST", "/api/chat", 200, _ollama_ok("done"))
    resp = OllamaLLM(stub.url, "m").complete("s", "p", max_tokens=8)
    assert resp.text == "done"
    assert len(stub.posts("/api/chat")) == 2
    assert no_sleep == [0.5]


def test_retry_429_is_transient(stub: Stub) -> None:
    stub.add("POST", "/v1/chat/completions", 429, {"error": "slow down"})
    stub.add("POST", "/v1/chat/completions", 200, _openai_ok())
    OpenAICompatLLM(f"{stub.url}/v1", "g", env={}).complete("s", "p", max_tokens=4)
    assert len(stub.posts("/v1/chat/completions")) == 2


def test_400_no_retry(stub: Stub, no_sleep: list[float]) -> None:
    stub.add("POST", "/api/chat", 400, {"error": "model 'm' not found"})
    with pytest.raises(LLMError) as ei:
        OllamaLLM(stub.url, "m").complete("s", "p", max_tokens=8)
    assert ei.value.status == 400
    assert "not found" in str(ei.value)
    assert len(stub.posts("/api/chat")) == 1
    assert no_sleep == []


def test_5xx_exhausts_three_attempts(stub: Stub, no_sleep: list[float]) -> None:
    for _ in range(3):
        stub.add("POST", "/api/chat", 500, {"error": "boom"})
    with pytest.raises(LLMError, match="3 attempts"):
        OllamaLLM(stub.url, "m").complete("s", "p", max_tokens=8)
    assert len(stub.posts("/api/chat")) == 3
    assert no_sleep == [0.5, 1.0]


def test_timeout_retried_then_error(stub: Stub) -> None:
    for _ in range(3):
        stub.add("POST", "/api/chat", 200, _ollama_ok(), delay=0.5)
    client = OllamaLLM(stub.url, "m", timeout_s=0.1)
    with pytest.raises(LLMError, match="timed out"):
        client.complete("s", "p", max_tokens=8)
    assert len(stub.posts("/api/chat")) == 3


def test_connection_refused_retried_then_error(stub: Stub) -> None:
    url = stub.url
    stub.server.shutdown()
    stub.server.server_close()
    with pytest.raises(LLMError, match="connection error"):
        OllamaLLM(url, "m").complete("s", "p", max_tokens=8)


def test_error_body_excerpt_truncated(stub: Stub) -> None:
    stub.add("POST", "/api/chat", 422, b"x" * 5000)
    with pytest.raises(LLMError) as ei:
        OllamaLLM(stub.url, "m").complete("s", "p", max_tokens=8)
    assert len(ei.value.body) == 500


# --------------------------------------------------------------------------- Budget


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_budget_precheck_rejects_over_cap() -> None:
    b = Budget(max_tokens_total=100)
    m = MeteredLLM(FakeLLM(["a"]), b, "n")
    with pytest.raises(BudgetExceeded, match="token budget"):
        m.complete("s", "p", max_tokens=101)
    assert b.used_tokens == 0
    assert m.records() == ()


def test_budget_accumulates_actual_tokens() -> None:
    b = Budget(max_tokens_total=1000)
    m = MeteredLLM(FakeLLM(["one two", "three"]), b, "n")
    m.complete("sys", "a b c", max_tokens=50)  # 1+3 in, 2 out
    m.complete("sys", "d", max_tokens=50)  # 1+1 in, 1 out
    assert b.used_tokens == 6 + 3
    assert b.remaining_tokens == 1000 - 9


def test_budget_check_uses_used_plus_max_tokens() -> None:
    b = Budget(max_tokens_total=20)
    m = MeteredLLM(FakeLLM(lambda s, p: "x"), b, "n")
    m.complete("s", "p", max_tokens=10)  # uses 3
    m.complete("s", "p", max_tokens=17)  # 3 + 17 = 20 -> allowed
    with pytest.raises(BudgetExceeded):
        m.complete("s", "p", max_tokens=15)  # 6 + 15 > 20


def test_budget_wall_clock_cap() -> None:
    clock = FakeClock()
    b = Budget(max_tokens_total=10_000, max_wall_s=60, clock=clock)
    m = MeteredLLM(FakeLLM(lambda s, p: "x"), b, "n")
    m.complete("s", "p", max_tokens=5)
    clock.t += 60.5
    assert b.elapsed_s == pytest.approx(60.5)
    with pytest.raises(BudgetExceeded, match="wall-clock"):
        m.complete("s", "p", max_tokens=5)


def test_budget_released_on_inner_failure() -> None:
    b = Budget(max_tokens_total=100)
    m = MeteredLLM(FakeLLM([]), b, "n")
    with pytest.raises(LLMError):
        m.complete("s", "p", max_tokens=100)
    # reservation released -> a full-cap call is allowed again
    MeteredLLM(FakeLLM(["ok"]), b, "n").complete("s", "p", max_tokens=100)
    assert b.used_tokens == 3


def test_budget_thread_safety_never_exceeds_cap() -> None:
    def slow(system: str, prompt: str) -> str:
        time.sleep(0.02)
        return "w " * 8  # 8 out tokens; 2 in tokens -> 10 per call

    b = Budget(max_tokens_total=50)
    m = MeteredLLM(FakeLLM(slow), b, "n")
    ok: list[int] = []
    rejected: list[int] = []
    barrier = threading.Barrier(20)

    def worker(i: int) -> None:
        barrier.wait()
        try:
            m.complete("s", "p", max_tokens=10)
            ok.append(i)
        except BudgetExceeded:
            rejected.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 5
    assert len(rejected) == 15
    assert b.used_tokens == 50 <= b.max_tokens_total
    assert len(m.records()) == 5


# --------------------------------------------------------------------------- MeteredLLM


def test_metered_records_hash_only() -> None:
    prompt = "very secret prompt text about the paper"
    m = MeteredLLM(FakeLLM(["answer here"]), Budget(), "subtasks")
    resp = m.complete("system msg", prompt, max_tokens=10)
    assert isinstance(m, LLM)
    (rec,) = m.records()
    assert rec.node == "subtasks"
    assert (rec.backend, rec.model) == ("fake", "fake")
    assert (rec.input_tokens, rec.output_tokens) == (resp.input_tokens, resp.output_tokens)
    expected = hashlib.sha256(f"system msg\x00{prompt}".encode()).hexdigest()
    assert rec.prompt_sha256 == expected
    dumped = rec.model_dump_json()
    assert prompt not in dumped
    assert "system msg" not in dumped
    assert set(rec.model_dump()) == {
        "node",
        "backend",
        "model",
        "input_tokens",
        "output_tokens",
        "latency_s",
        "prompt_sha256",
    }


def test_metered_records_returns_copy() -> None:
    m = MeteredLLM(FakeLLM(["a"]), Budget(), "n")
    m.complete("s", "p", max_tokens=5)
    snap = m.records()
    assert isinstance(snap, tuple) and len(snap) == 1


# --------------------------------------------------------------------------- FakeLLM


def test_fake_scripted() -> None:
    f = FakeLLM(["first answer", "second"])
    r1 = f.complete("sys a", "p q r", max_tokens=5)
    r2 = f.complete("s", "x", max_tokens=5)
    assert (r1.text, r1.input_tokens, r1.output_tokens) == ("first answer", 5, 2)
    assert r2.text == "second"
    assert r1.backend == "fake"
    assert f.prompts == [("sys a", "p q r"), ("s", "x")]
    with pytest.raises(LLMError, match="exhausted"):
        f.complete("s", "x", max_tokens=5)


def test_fake_callable_is_deterministic() -> None:
    f = FakeLLM(lambda s, p: f"echo {p}")
    assert f.complete("s", "hi", max_tokens=5).text == "echo hi"
    assert f.complete("s", "hi", max_tokens=5).text == "echo hi"
    assert len(f.prompts) == 2


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("ollama:qwen2.5-coder:7b", ("ollama", "qwen2.5-coder:7b")),
        ("openai:gpt-4o-mini", ("openai", "gpt-4o-mini")),
        ("fake:x", ("fake", "x")),
        ("ollama:hf.co/org/model:Q4_K_M", ("ollama", "hf.co/org/model:Q4_K_M")),
    ],
)
def test_parse_spec(spec: str, expected: tuple[str, str]) -> None:
    assert parse_spec(spec) == expected


@pytest.mark.parametrize("bad", ["anthropic:claude", "ollama", "ollama:", "", ":model"])
def test_parse_spec_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_spec(bad)


def test_make_llm_ollama_defaults_and_env() -> None:
    d = make_llm("ollama:qwen2.5-coder:7b", {})
    assert isinstance(d, OllamaLLM)
    assert (d.host, d.model, d.num_ctx) == ("http://localhost:11434", "qwen2.5-coder:7b", 8192)
    c = make_llm("ollama:m", {"OLLAMA_HOST": "gpu-box:11434", "OLLAMA_NUM_CTX": "4096"})
    assert isinstance(c, OllamaLLM)
    assert (c.host, c.num_ctx) == ("http://gpu-box:11434", 4096)
    with pytest.raises(ValueError, match="OLLAMA_NUM_CTX"):
        make_llm("ollama:m", {"OLLAMA_NUM_CTX": "big"})


def test_make_llm_openai_and_fake() -> None:
    o = make_llm("openai:gpt-4o-mini", {})
    assert isinstance(o, OpenAICompatLLM)
    assert o.base_url == "https://api.openai.com/v1"
    assert isinstance(make_llm("fake:any", {}), FakeLLM)


D30_MSG = "use the native 'ollama:' backend (num_ctx cannot be set via the OpenAI-compatible"


@pytest.mark.parametrize(
    "env",
    [
        {"OPENAI_BASE_URL": "http://localhost:11434/v1"},
        {"OPENAI_BASE_URL": "http://127.0.0.1:11434/v1", "OLLAMA_HOST": "gpu-box:9999"},
        {"OPENAI_BASE_URL": "http://gpu-box:8080/v1", "OLLAMA_HOST": "gpu-box:8080"},
        {"OPENAI_BASE_URL": "http://localhost:11434/v1", "REPRODUCER_ALLOW_OLLAMA_COMPAT": "true"},
    ],
)
def test_make_llm_openai_rejects_ollama_endpoint(env: dict[str, str]) -> None:
    with pytest.raises(ValueError) as ei:
        make_llm("openai:qwen2.5-coder:7b", env)
    assert D30_MSG in str(ei.value)


def test_make_llm_openai_ollama_allowed_with_flag() -> None:
    env = {"OPENAI_BASE_URL": "http://localhost:11434/v1", "REPRODUCER_ALLOW_OLLAMA_COMPAT": "1"}
    o = make_llm("openai:qwen2.5-coder:7b", env)
    assert isinstance(o, OpenAICompatLLM)
    assert (o.base_url, o.model) == ("http://localhost:11434/v1", "qwen2.5-coder:7b")


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://localhost:8000/v1"])
def test_make_llm_openai_non_ollama_url_allowed(url: str) -> None:
    o = make_llm("openai:m", {"OPENAI_BASE_URL": url})
    assert isinstance(o, OpenAICompatLLM)
    assert o.base_url == url


def test_llm_for_node_override_and_shared_budget() -> None:
    env = {
        "REPRODUCER_LLM": "ollama:qwen2.5-coder:7b",
        "REPRODUCER_LLM_CRITIC": "openai:gpt-4o-mini",
    }
    budget = Budget()
    critic = llm_for_node("critic", budget, env)
    subtasks = llm_for_node("subtasks", budget, env)
    assert isinstance(critic.inner, OpenAICompatLLM)
    assert critic.inner.model == "gpt-4o-mini"
    assert isinstance(subtasks.inner, OllamaLLM)
    assert subtasks.inner.model == "qwen2.5-coder:7b"
    assert critic.budget is subtasks.budget is budget
    assert (critic.node, subtasks.node) == ("critic", "subtasks")


def test_llm_for_node_shared_budget_accumulates_across_nodes() -> None:
    budget = Budget(max_tokens_total=1000)
    env = {"REPRODUCER_LLM": "fake:x"}
    llm_for_node("a", budget, env).complete("one", "two", max_tokens=5)
    llm_for_node("b", budget, env).complete("one", "two", max_tokens=5)
    assert budget.used_tokens == 2 * (2 + 1)


def test_llm_for_node_missing_config_is_clear() -> None:
    with pytest.raises(ValueError) as ei:
        llm_for_node("reflect", Budget(), {})
    msg = str(ei.value)
    assert "REPRODUCER_LLM_REFLECT" in msg and "REPRODUCER_LLM" in msg


def test_llm_for_node_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPRODUCER_LLM_ORCHESTRATOR", raising=False)
    monkeypatch.setenv("REPRODUCER_LLM", "fake:x")
    assert isinstance(llm_for_node("orchestrator", Budget()).inner, FakeLLM)


# --------------------------------------------------------------------------- live


LIVE_MODEL = os.environ.get("REPRODUCER_LIVE_MODEL", "qwen2.5-coder:7b")
LIVE_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if "://" not in LIVE_HOST:
    LIVE_HOST = f"http://{LIVE_HOST}"


def _require_local_model() -> None:
    try:
        with urllib.request.urlopen(f"{LIVE_HOST}/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
    except OSError as e:
        pytest.skip(f"Ollama not reachable at {LIVE_HOST}: {e}")
    names = {m.get("name") for m in tags.get("models", [])}
    if LIVE_MODEL not in names:
        pytest.skip(
            f"model {LIVE_MODEL!r} not pulled (have {sorted(names)}); "
            f"run `ollama pull {LIVE_MODEL}` or set REPRODUCER_LIVE_MODEL"
        )


def _check_schema_json(resp: LLMResponse, label: str) -> None:
    print(
        f"\n[{label}] model={resp.model} latency={resp.latency_s:.2f}s "
        f"in={resp.input_tokens} out={resp.output_tokens} text={resp.text!r}"
    )
    obj = json.loads(resp.text)
    assert isinstance(obj, dict) and isinstance(obj.get("ok"), bool)
    assert resp.input_tokens > 0 and resp.output_tokens > 0


LIVE_SYSTEM = "You are a JSON API. Reply only with JSON matching the schema."
LIVE_PROMPT = 'Is 2+2 equal to 4? Answer as {"ok": <boolean>}.'


@pytest.mark.llm_live
def test_live_ollama_native_schema() -> None:
    _require_local_model()
    client = OllamaLLM(LIVE_HOST, LIVE_MODEL)
    resp = client.complete(LIVE_SYSTEM, LIVE_PROMPT, max_tokens=64, json_schema=SCHEMA)
    _check_schema_json(resp, "ollama-native")


@pytest.mark.llm_live
def test_live_ollama_openai_compat_schema() -> None:
    _require_local_model()
    env = {
        "OPENAI_BASE_URL": f"{LIVE_HOST}/v1",
        "OLLAMA_HOST": LIVE_HOST,
        "REPRODUCER_ALLOW_OLLAMA_COMPAT": "1",  # D30 escape hatch, live compat test only
    }
    client = make_llm(f"openai:{LIVE_MODEL}", env)
    resp = client.complete(LIVE_SYSTEM, LIVE_PROMPT, max_tokens=64, json_schema=SCHEMA)
    _check_schema_json(resp, "ollama-openai-compat")


@pytest.mark.llm_live
def test_live_ollama_unload() -> None:
    _require_local_model()
    assert OllamaLLM(LIVE_HOST, LIVE_MODEL).unload() is True


@pytest.mark.llm_live
def test_live_openai_api_schema() -> None:
    model = os.environ.get("REPRODUCER_LIVE_API_MODEL")
    if not os.environ.get("OPENAI_API_KEY") or not model:
        pytest.skip("set OPENAI_API_KEY and REPRODUCER_LIVE_API_MODEL to run")
    base_url = os.environ.get("OPENAI_BASE_URL", llm_mod.DEFAULT_OPENAI_BASE_URL)
    client = OpenAICompatLLM(base_url, model)
    resp = client.complete(LIVE_SYSTEM, LIVE_PROMPT, max_tokens=64, json_schema=SCHEMA)
    _check_schema_json(resp, "openai-api")
