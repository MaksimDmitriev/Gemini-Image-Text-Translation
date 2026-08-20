"""Provider factory."""

from drawing_translator.models import RegionProvider


BACKENDS = ("gemini", "google-openai")


def create_provider(backend: str, model: str | None = None) -> RegionProvider:
    if backend == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(model=model or GeminiProvider.default_model)
    if backend == "google-openai":
        from .google_openai import GoogleOpenAIProvider

        return GoogleOpenAIProvider(model=model or GoogleOpenAIProvider.default_model)
    raise ValueError(f"unknown backend {backend!r}; choose one of {', '.join(BACKENDS)}")


__all__ = ["BACKENDS", "create_provider"]
