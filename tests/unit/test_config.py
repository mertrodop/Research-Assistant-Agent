from ra_agent.config import Settings
from ra_agent.llm.client import NullVisionClient, build_llm


def test_model_roles_load_from_yaml():
    s = Settings.load("config/pipeline.yaml")
    assert s.models["agent"].model == "qwen/qwen3-vl-8b-instruct"
    assert s.models["vision"].base_url == "https://openrouter.ai/api/v1"
    assert s.agent.max_steps == 12 and s.paths.vault_dir == "vault"


def test_missing_key_gives_null_client(monkeypatch):
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    assert isinstance(build_llm(Settings().models["vision"]), NullVisionClient)
