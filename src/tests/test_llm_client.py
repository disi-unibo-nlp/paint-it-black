import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.llm_client import LLMClient


def _mock_openai_client(response_text="hello"):
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return mock_client


def test_llm_client_complete_returns_string():
    with patch("core.llm_client.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai_client("model reply")
        client = LLMClient("http://localhost/v1", "key", "model-x")
        result = client.complete([{"role": "user", "content": "hi"}])
    assert result == "model reply"


def test_llm_client_passes_messages():
    with patch("core.llm_client.openai.OpenAI") as MockOpenAI:
        mock = _mock_openai_client()
        MockOpenAI.return_value = mock
        client = LLMClient("http://localhost/v1", "key", "model-x", max_new_tokens=512)
        msgs = [{"role": "user", "content": "test"}]
        client.complete(msgs)
    call_kwargs = mock.chat.completions.create.call_args
    assert call_kwargs.kwargs["messages"] == msgs
    assert call_kwargs.kwargs["max_tokens"] == 512


def test_llm_client_retry_on_failure():
    with patch("core.llm_client.openai.OpenAI") as MockOpenAI, \
         patch("core.llm_client.time.sleep"):
        mock = _mock_openai_client("ok")
        mock.chat.completions.create.side_effect = [
            RuntimeError("timeout"),
            mock.chat.completions.create.return_value,
        ]
        MockOpenAI.return_value = mock
        client = LLMClient("http://localhost/v1", "key", "m", max_retries=3)
        result = client.complete([])
    assert result == "ok"
    assert mock.chat.completions.create.call_count == 2


def test_llm_client_raises_after_max_retries():
    with patch("core.llm_client.openai.OpenAI") as MockOpenAI, \
         patch("core.llm_client.time.sleep"):
        mock = MagicMock()
        mock.chat.completions.create.side_effect = RuntimeError("always fails")
        MockOpenAI.return_value = mock
        client = LLMClient("http://localhost/v1", "key", "m", max_retries=2)
        with pytest.raises(RuntimeError, match="always fails"):
            client.complete([])
    assert mock.chat.completions.create.call_count == 2
