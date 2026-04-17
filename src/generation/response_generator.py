"""LLM response generation logic supporting Anthropic API and HuggingFace fallback."""

import os
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from src.generation.prompt_templates import get_template, format_user_prompt

load_dotenv()


class ResponseGenerator:
    """Generates customer support responses using an LLM backend.

    Supports two backends controlled by the 'provider' config key:
    - 'anthropic': Uses the Anthropic Claude API (preferred).
    - 'huggingface': Uses a local HuggingFace pipeline as fallback.

    Args:
        cfg: Full config dict loaded from config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        gc = cfg["generation"]
        self.provider: str = gc.get("provider", "anthropic")
        self.max_tokens: int = gc.get("max_tokens", 300)
        self.temperature: float = gc.get("temperature", 0.3)
        self.top_p: float = gc.get("top_p", 0.9)

        if self.provider == "anthropic":
            self._init_anthropic(gc)
        else:
            self._init_huggingface(gc)

    def _init_anthropic(self, gc: dict) -> None:
        try:
            import anthropic  # type: ignore

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Create a .env file with your key or set it in the environment."
                )
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name: str = gc.get("model", "claude-haiku-4-5-20251001")
            logger.info(f"Anthropic provider initialised with model '{self.model_name}'.")
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e

    def _init_huggingface(self, gc: dict) -> None:
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
            import torch

            model_name = gc.get("hf_model", "mistralai/Mistral-7B-Instruct-v0.2")
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading HuggingFace model '{model_name}' (device={device})…")
            self._hf_pipe = hf_pipeline(
                "text-generation",
                model=model_name,
                device=device,
                torch_dtype="auto",
            )
            self.model_name = model_name
            logger.info("HuggingFace pipeline ready.")
        except ImportError as e:
            raise ImportError(
                "transformers package not installed. Run: pip install transformers"
            ) from e

    def generate(self, query: str, intent: str) -> tuple:
        """Generate a support response and return (response_text, context_used)."""
        template = get_template(intent)
        system_msg = template["system"]
        user_msg = format_user_prompt(intent, query)
        context = system_msg + "\n\n" + user_msg

        if self.provider == "anthropic":
            response = self._generate_anthropic(system_msg, user_msg)
        else:
            response = self._generate_huggingface(system_msg, user_msg)

        return response, context

    def _generate_anthropic(self, system_msg: str, user_msg: str) -> str:
        """Call Anthropic Messages API and return the response text."""
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

    def _generate_huggingface(self, system_msg: str, user_msg: str) -> str:
        """Call HuggingFace pipeline with instruction format and return the response text."""
        prompt = (
            f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"
        )
        try:
            outputs = self._hf_pipe(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_full_text=False,
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            raise
