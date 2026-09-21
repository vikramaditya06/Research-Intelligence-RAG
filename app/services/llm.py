import json
import math
import time

from openai import (
    OpenAI,
    BadRequestError,
    APIError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)

from app.core.config import settings


def client():
    kwargs = {"api_key": settings.openai_api_key}

    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return OpenAI(**kwargs)


def _batched(items, size):
    if size <= 0:
        raise ValueError("batch size must be positive")

    for i in range(0, len(items), size):
        yield items[i:i + size]


def _with_retry(operation, attempts=3):
    for attempt in range(attempts):
        try:
            return operation()

        except BadRequestError:
            raise

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            APIError,
        ):
            if attempt == attempts - 1:
                raise

            time.sleep(2 ** attempt)

    raise RuntimeError("unreachable")


def embed_batches(texts, batch_size: int | None = None):
    """Yield embedding batches without retaining the entire input/output set."""

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    size = batch_size or settings.embedding_batch_size

    if size <= 0:
        raise ValueError("batch size must be positive")

    api_client = client()
    batch = []

    for text in texts:
        batch.append(text)

        if len(batch) >= size:
            yield _embed_batch(api_client, batch)
            batch = []

    if batch:
        yield _embed_batch(api_client, batch)


def _fit_gemini_embedding(vector: list[float]) -> list[float]:
    """
    Gemini Embedding 001 normally returns 3072 dimensions.
    Our pgvector schema uses 1536 dimensions.

    Gemini supports smaller MRL dimensions. For gemini-embedding-001,
    the reduced vector must be normalized manually.
    """

    expected = settings.embedding_dimension
    actual = len(vector)

    if actual == expected:
        return vector

    model_name = settings.embedding_model.lower()

    if (
        model_name == "gemini-embedding-001"
        and actual > expected
    ):
        reduced = vector[:expected]

        norm = math.sqrt(
            sum(value * value for value in reduced)
        )

        if norm == 0:
            raise RuntimeError(
                "Gemini embedding has zero norm after dimensionality reduction"
            )

        return [value / norm for value in reduced]

    raise RuntimeError(
        f"Embedding dimension mismatch: "
        f"expected {expected}, got {actual}"
    )


def _embed_batch(api_client, batch: list[str]) -> list[list[float]]:
    response = _with_retry(
        lambda: api_client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
    )

    # Gemini's OpenAI-compatible endpoint may return index=None.
    # Preserve response order when indexes are unavailable.
    data = list(response.data)

    if data and all(item.index is not None for item in data):
        data.sort(key=lambda x: x.index)

    vectors = [x.embedding for x in data]

    if len(vectors) != len(batch):
        raise RuntimeError(
            "Embedding response count does not match input batch"
        )

    vectors = [
        _fit_gemini_embedding(vector)
        for vector in vectors
    ]

    return vectors


def embed(texts: list[str]) -> list[list[float]]:
    return [
        vector
        for batch in embed_batches(texts)
        for vector in batch
    ]


def generate(prompt: str, system: str | None = None) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    api_client = client()

    response = _with_retry(
        lambda: api_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": system
                    or "You are a rigorous research assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
    )

    return response.choices[0].message.content or ""


def _response_format_unsupported(exc: BadRequestError) -> bool:
    """Return True only for errors indicating JSON response-format incompatibility."""

    message = str(exc).lower()

    return (
        "response_format" in message
        or "json_object" in message
        or (
            "json mode" in message
            and (
                "support" in message
                or "unsupported" in message
            )
        )
        or (
            "structured output" in message
            and (
                "support" in message
                or "unsupported" in message
            )
        )
    )


def generate_json(
    prompt: str,
    system: str | None = None,
) -> dict:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    api_client = client()

    # First try the OpenAI-compatible JSON-object mode.
    try:
        response = _with_retry(
            lambda: api_client.chat.completions.create(
                model=settings.llm_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": system or "Return valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
        )

        text = response.choices[0].message.content or "{}"

    except BadRequestError as exc:
        if not _response_format_unsupported(exc):
            raise
        # Gemini's OpenAI-compatible layer can differ from OpenAI on
        # response-format details. Fall back to plain text generation
        # and parse the JSON ourselves.
        text = generate(
            prompt + "\nReturn ONLY a valid JSON object. No markdown.",
            system=system,
        )

    # Remove accidental markdown fences.
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {text[:500]}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("LLM JSON response must be an object")

    return data