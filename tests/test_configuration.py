import pytest

from provenance_graphrag import config


def test_importing_algorithms_does_not_require_credentials() -> None:
    assert config.NEO4J_USER


def test_missing_credentials_fail_at_the_network_boundary(monkeypatch) -> None:
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        config.require_credentials("GEMINI_API_KEY")
