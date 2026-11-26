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


def test_yaml_with_one_role_keeps_the_other(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("models:\n  agent:\n    model: local-model\n")
    s = Settings.load(path)
    assert s.models["agent"].model == "local-model"
    assert s.models["vision"].model == "qwen/qwen3-vl-8b-instruct"


def test_local_endpoint_without_key_gets_placeholder(monkeypatch):
    from ra_agent.config import ModelSettings
    from ra_agent.llm.client import OpenAICompatibleVisionClient, resolve_api_key

    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    local = ModelSettings(base_url="http://localhost:11434/v1")
    assert resolve_api_key(local) == "not-needed"
    assert isinstance(build_llm(local), OpenAICompatibleVisionClient)
    assert resolve_api_key(ModelSettings()) == ""


def test_load_none_reads_repo_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # found from the repo root, not cwd
    assert Settings.load().models["agent"].model == "qwen/qwen3-vl-8b-instruct"
