import logging
import time
from typing import Optional

import openai

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin OpenAI-compatible chat client with retry logic."""

    def __init__(
        self,
        base_url,
        api_key,
        model,
        max_new_tokens=2048,
        timeout=120,
        max_retries=3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

        # Standard OpenAI sampling params
        self._temperature = temperature
        self._top_p = top_p
        # vLLM extensions (passed via extra_body)
        self._top_k = top_k
        self._min_p = min_p
        self._enable_thinking = enable_thinking

    def complete(self, messages: list[dict]) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_new_tokens,
        )
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p

        extra: dict = {}
        if self._top_k is not None:
            extra["top_k"] = self._top_k
        if self._min_p is not None:
            extra["min_p"] = self._min_p
        if self._enable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": True}
        if extra:
            kwargs["extra_body"] = extra

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("LLM call failed (attempt %d/%d): %r. Retrying in %ds...",
                               attempt + 1, self.max_retries, exc, wait)
                time.sleep(wait)
        raise last_exc
