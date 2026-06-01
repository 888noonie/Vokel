from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_provider_specific_modules_are_not_bundled() -> None:
    for module_name in ("web_search.py", "image_search.py", "giphy_search.py"):
        assert not (ROOT / "src" / "vokel" / module_name).exists()


def test_env_template_does_not_request_provider_api_keys() -> None:
    env_template = (ROOT / ".env.example").read_text()
    for variable_name in ("SERPAPI_API_KEY", "UNSPLASH_ACCESS_KEY", "GIPHY_API_KEY"):
        assert variable_name not in env_template
