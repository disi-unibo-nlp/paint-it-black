import logging
import time

import openai

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin OpenAI-compatible chat client with retry logic."""

    def __init__(self, base_url, api_key, model, max_new_tokens=2048, timeout=120, max_retries=3):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def complete(self, messages: list[dict]) -> str:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("LLM call failed (attempt %d/%d): %r. Retrying in %ds...",
                               attempt + 1, self.max_retries, exc, wait)
                time.sleep(wait)
        raise last_exc
