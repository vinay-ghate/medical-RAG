"""LLM provider factory supporting Groq and local Ollama."""

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from src.config import settings, setup_logger

logger = setup_logger("generation.llm")


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Instantiate and return the configured chat language model."""
    active_provider = (provider or settings.llm_provider).lower()
    temp = settings.temperature if temperature is None else temperature

    if active_provider == "groq":
        api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "GROQ_API_KEY is not set. Please set it in your .env file or environment."
            )
        model = model_name or settings.groq_model
        logger.info("Initializing Groq LLM with model '%s' (temperature=%s)", model, temp)
        return ChatGroq(
            model=model,
            temperature=temp,
            api_key=api_key,
            max_retries=2,
        )

    elif active_provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        model = model_name or settings.ollama_model
        logger.info("Initializing local Ollama LLM with model '%s'", model)
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=temp,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{active_provider}'. Supported options: 'groq', 'ollama'."
        )
