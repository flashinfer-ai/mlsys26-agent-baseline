import logging
import os
import random
import time

import anthropic
import openai

logger = logging.getLogger(__name__)


def _require_env(var_names: list[str], api_type: str) -> str:
    """Return the first configured env var value or raise a clear error."""
    for var_name in var_names:
        value = os.environ.get(var_name)
        if value:
            return value

    raise RuntimeError(
        f"Missing credentials for api_type='{api_type}'. Set one of: "
        + ", ".join(var_names)
    )


def create_inference_server(api_type: str):
    """Create an LLM client based on API type."""
    if api_type == "openai":
        return openai.OpenAI(
            api_key=_require_env(["OPENAI_API_KEY"], api_type),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
    elif api_type in ("claude", "anthropic"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if not api_key and not auth_token:
            _require_env(["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"], api_type)
        return anthropic.Anthropic(api_key=api_key, auth_token=auth_token)
    elif api_type in ("gemini", "google"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            _require_env(["GEMINI_API_KEY"], api_type)
        return openai.OpenAI(
            api_key=_require_env(["GEMINI_API_KEY"], api_type),
            base_url=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        )
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")


def _query_openai(client, model_name, prompt, max_completion_tokens, **kwargs):
    """Query OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_completion_tokens,
        **kwargs,
    )
    return response.choices[0].message.content


def _query_anthropic(client, model_name, prompt, max_completion_tokens, **kwargs):
    """Query Anthropic API directly."""
    response = client.messages.create(
        model=model_name,
        max_tokens=max_completion_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return "".join(b.text for b in response.content if hasattr(b, "text"))


def query_inference_server(
    server,
    model_name: str,
    prompt: str,
    max_completion_tokens: int = 16384,
    retry_times: int = 5,
    **kwargs,
):
    """Query LLM with retry and exponential backoff."""
    kwargs.setdefault("temperature", 1.0)
    is_anthropic = isinstance(server, anthropic.Anthropic)
    query_fn = _query_anthropic if is_anthropic else _query_openai

    for attempt in range(retry_times):
        try:
            return query_fn(server, model_name, prompt, max_completion_tokens, **kwargs)
        except Exception as e:
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{retry_times}): {e}"
            )
            if attempt == retry_times - 1:
                raise
            wait_time = (2**attempt) + random.uniform(0, 1)
            logger.info(f"Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)
