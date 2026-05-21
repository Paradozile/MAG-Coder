from pathlib import Path

from magcoder.generation import (
    extract_python_code,
    extract_python_code_with_diagnostics,
    prompt_hash,
)
from magcoder.generation.openai_compatible import extract_openai_compatible_text
from magcoder.generation.prompt_profile import load_prompt_profile
from magcoder.generation.prompt_wrappers import apply_prompt_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "configs" / "prompt_profiles"


def test_prompt_profiles_can_load() -> None:
    for path in sorted(PROFILE_DIR.glob("*.yaml")):
        profile = load_prompt_profile(path)
        assert profile.profile_id
        assert profile.system_message
        assert profile.require_final_content


def test_compact_contract_profile_requests_begin_end_python() -> None:
    profile = load_prompt_profile(PROFILE_DIR / "ollama_qwen_contract_compact.yaml")
    prompt = "Implement the required contract."

    wrapped = apply_prompt_profile(prompt, profile)

    assert profile.extraction_strategy == "begin_end_python"
    assert "BEGIN_PYTHON" in wrapped
    assert "END_PYTHON" in wrapped
    assert "Each required interaction method must call self.send_message" in wrapped


def test_apply_prompt_profile_changes_prompt() -> None:
    profile = load_prompt_profile(PROFILE_DIR / "ollama_qwen_code.yaml")
    prompt = "Generate code for this MAG."

    wrapped = apply_prompt_profile(prompt, profile)

    assert prompt in wrapped
    assert wrapped != prompt
    assert "Do not write reasoning" in wrapped
    assert "Implement every required method name exactly as specified" in wrapped


def test_prompt_hash_changes_when_profile_is_applied() -> None:
    profile = load_prompt_profile(PROFILE_DIR / "ollama_qwen_code.yaml")
    prompt = "Generate code for this MAG."

    assert prompt_hash(prompt) != prompt_hash(apply_prompt_profile(prompt, profile))


def test_extraction_supports_fenced_code() -> None:
    assert extract_python_code("```python\ndef add(a, b):\n    return a + b\n```") == (
        "def add(a, b):\n    return a + b\n"
    )


def test_extraction_supports_begin_end_python() -> None:
    result = extract_python_code_with_diagnostics(
        "BEGIN_PYTHON\ndef add(a, b):\n    return a + b\nEND_PYTHON"
    )

    assert result.extraction_strategy_used == "begin_end_python"
    assert result.code.startswith("def add")


def test_extraction_selects_parseable_code_block() -> None:
    result = extract_python_code_with_diagnostics(
        "```python\ndef broken(:\n```\n\n"
        "```python\ndef add(a, b):\n    return a + b\n```"
    )

    assert result.extraction_strategy_used == "python_code_fence"
    assert result.code == "def add(a, b):\n    return a + b\n"


def test_extraction_supports_json_code_field() -> None:
    result = extract_python_code_with_diagnostics(
        '{"code": "def add(a, b):\\n    return a + b\\n"}'
    )

    assert result.extraction_strategy_used == "json_code_field"
    assert result.code == "def add(a, b):\n    return a + b\n"


def test_extraction_supports_bare_class_or_def_code() -> None:
    result = extract_python_code_with_diagnostics(
        "class Calculator:\n    def add(self, a, b):\n        return a + b"
    )

    assert result.extraction_strategy_used == "bare_python"
    assert result.extracted_code_non_empty


def test_final_content_empty_does_not_extract_reasoning_code() -> None:
    response = (
        '{"choices":[{"message":{"content":"",'
        '"reasoning":"def add(a,b): return a+b"}}]}'
    )
    extraction = extract_openai_compatible_text(response)
    code = extract_python_code_with_diagnostics(extraction.text)

    assert extraction.text == ""
    assert code.code == ""
    assert code.extraction_strategy_used == "empty_final_content"
    assert extraction.schema_summary["reasoning_present"] is True
