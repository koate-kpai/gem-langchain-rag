# conftest.py
"""
Shared fixtures and configuration for the test suite.

Testing Philosophy:
  ALL external API calls are mocked. This means:
    - Tests run with zero OpenAI cost
    - Tests run offline (no network required)
    - Tests complete in <1 second
    - Tests are reproducible (no dependency on API behavior)

  This follows the "Fake It Till You Make It" pattern for testing ML pipelines
  — mock the API boundary, test the business logic thoroughly.

  Note: pytest automatically discovers conftest.py in the tests/ directory.
  Fixtures defined here are available to all test files without explicit import.
"""
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: build a proper mock response for OpenAI error constructors
# ---------------------------------------------------------------------------
def make_mock_response(status_code: int = 429) -> MagicMock:
    """Create a mock HTTP response object for OpenAI error constructors.

    The openai library v1+ requires 'response' and 'body' keyword arguments
    for all APIStatusError subclasses. We build minimal mock objects that
    satisfy the constructor signature.
    """
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    response.request = MagicMock()
    return response


# ---------------------------------------------------------------------------
# Fixtures: data directory with test documents
# ---------------------------------------------------------------------------
@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a sample .txt document."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    doc = data_dir / "policy.txt"
    doc.write_text(
        "Remote Work Policy\n"
        "Employees are allowed to work remotely up to 3 days a week.\n"
    )
    return data_dir


@pytest.fixture
def test_empty_dir(tmp_path: Path) -> Path:
    """Create an empty data directory (no documents)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# ---------------------------------------------------------------------------
# Fixtures: mock OpenAI to prevent real API calls during tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_openai_api() -> Generator[MagicMock, None, None]:
    """Mock all OpenAI API calls.

    This fixture is auto-used (applies to every test). It patches:
      - openai.resources.Embeddings.create
      - openai.resources.chat.completions.Completions.create

    The embedding mock dynamically determines how many vectors to return
    based on the length of the input (handles strings, lists of strings,
    and token arrays).
    """
    with patch("openai.resources.Embeddings.create") as mock_embed:
        def _embed_side_effect(*args, **kwargs):
            inp = kwargs.get("input", args[0] if args else [])
            # Determine batch size: could be string, list[str], or list[list[int]]
            if isinstance(inp, str):
                batch_size = 1
            elif isinstance(inp, list):
                batch_size = len(inp)
            else:
                batch_size = 1
            mock_resp = MagicMock()
            mock_resp.data = [
                MagicMock(embedding=[float(i + 1) / 1536] * 1536, index=i)
                for i in range(batch_size)
            ]
            mock_resp.model = "text-embedding-3-small"
            mock_resp.usage = MagicMock(prompt_tokens=batch_size * 10)
            return mock_resp
        mock_embed.side_effect = _embed_side_effect
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat:
            mock_chat.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Mocked answer."))]
            )
            yield mock_embed


# ---------------------------------------------------------------------------
# Fixture: mock environment to provide a real-looking API key
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def set_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a valid-looking OpenAI API key for all tests.

    This prevents _validate_api_key from raising during tests.
    """
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "sk-proj-TestKeyThatLooksRealButIsNotActuallyValid12345",
    )


# ---------------------------------------------------------------------------
# Fixture: mock .env file so path existence checks pass
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_env_file(tmp_path: Path) -> Path:
    """Create a temporary .env file at config.paths.env_file location.

    Note: we cannot use monkeypatch.setattr on frozen dataclass fields.
    Instead, we create the file at the location the config already points to
    during test execution by temporarily overriding via a config module patch.
    """
    import config as cfg_module
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        'OPENAI_API_KEY="sk-proj-MockedKeyForTestingPurposesOnly789"\n'
    )
    # Use object.__setattr__ to bypass frozen dataclass restriction
    object.__setattr__(cfg_module.config.paths, "env_file", env_file)
    yield env_file
    # Restore original path after test
    from config import Paths
    object.__setattr__(
        cfg_module.config.paths,
        "env_file",
        cfg_module._BASE_DIR / ".env",
    )
