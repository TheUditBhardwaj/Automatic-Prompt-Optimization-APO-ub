import os
import time
import re
import logging
import json
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Type, Optional, Tuple, Any, List, Dict
from pydantic import BaseModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default model names
DEFAULT_MODEL = "gemini-2.5-flash"
FLASH_MODEL = "gemini-2.5-flash"


def compute_gemini_cost(prompt_tokens: int, candidate_tokens: int, model_name: str) -> float:
    """Computes cost based on Google Gemini API pricing."""
    is_pro = "pro" in model_name.lower()
    input_rate = 1.25 / 1_000_000 if is_pro else 0.075 / 1_000_000
    output_rate = 5.00 / 1_000_000 if is_pro else 0.30 / 1_000_000
    return (prompt_tokens * input_rate) + (candidate_tokens * output_rate)


def execute_with_retry(func, *args, max_attempts=6, base_delay=5.0, **kwargs) -> Any:
    """Wraps API calls with rate-limit-aware exponential backoff retry logic."""
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            is_retryable = (
                "429" in err_msg or
                "resource_exhausted" in err_msg.lower() or
                "503" in err_msg or
                "quota" in err_msg.lower() or
                "500" in err_msg or
                "unavailable" in err_msg.lower()
            )

            # Check if the daily quota is exhausted (GenerateRequestsPerDay limit reached)
            is_daily_limit = "perday" in err_msg.lower() or "daily" in err_msg.lower()
            if is_daily_limit:
                logger.warning("Daily API quota reached. Sleeping for 1 hour before retrying...")
                time.sleep(3600)  # Sleep for 1 hour and try again
                continue

            if is_retryable and attempt < max_attempts:
                wait_time = base_delay * (2 ** (attempt - 1))
                # Try to parse retry delay from the error message
                match = re.search(r"retry in ([\d\.]+)s", err_msg, re.IGNORECASE)
                if match:
                    try:
                        wait_time = float(match.group(1)) + 1.5
                    except ValueError:
                        pass
                wait_time = min(wait_time, 120.0)
                logger.warning(
                    f"Gemini API error on attempt {attempt}/{max_attempts}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                raise


class GeminiClient:
    """Wrapper around the Google Gemini API (Gemini 3.1 Pro) with automatic retries and cost tracking."""

    def __init__(self, api_key: str = None):
        self.api_keys = []
        primary = api_key or os.environ.get("GEMINI_API_KEY")
        if primary:
            self.api_keys.append(primary)

        # Load GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. dynamically
        for k, v in os.environ.items():
            if k.startswith("GEMINI_API_KEY_") and v.strip():
                self.api_keys.append(v.strip())

        # Remove duplicate keys while keeping order
        self.api_keys = list(dict.fromkeys(self.api_keys))

        if not self.api_keys:
            logger.warning("No API keys found starting with GEMINI_API_KEY.")

        self.clients = [genai.Client(api_key=key) for key in self.api_keys]
        self.cooldowns = {}  # key -> timestamp until cooled down
        self.current_key_index = 0

    def execute_with_retry_and_rotation(self, func, max_attempts_per_key=3, base_delay=5.0) -> Any:
        """Executes API call using the active client and rotates keys on rate limit/quota failure."""
        tried_keys = set()
        total_keys = len(self.api_keys)

        while len(tried_keys) < total_keys or not total_keys:
            now = time.time()
            active_key = None
            active_client = None

            # Find an active key not on cooldown
            for _ in range(total_keys):
                key = self.api_keys[self.current_key_index]
                cooldown_until = self.cooldowns.get(key, 0.0)
                client = self.clients[self.current_key_index]

                # Move round-robin index forward
                self.current_key_index = (self.current_key_index + 1) % total_keys

                if now >= cooldown_until:
                    active_key = key
                    active_client = client
                    break

            # If all keys on cooldown, pick the one nearest to completion and sleep
            if not active_key and total_keys > 0:
                earliest_key = min(self.api_keys, key=lambda k: self.cooldowns.get(k, 0.0))
                idx = self.api_keys.index(earliest_key)
                active_key = earliest_key
                active_client = self.clients[idx]
                wait_duration = max(0.1, self.cooldowns.get(active_key, 0.0) - now)
                logger.warning(f"All API keys are in cooldown. Waiting {wait_duration:.1f}s for earliest key...")
                time.sleep(wait_duration)

            if not active_client:
                raise ValueError("No Gemini API keys loaded. Please check your environment variables.")

            masked_key = f"...{active_key[-6:]}" if len(active_key) > 6 else "key"
            logger.info(f"Using active API key: {masked_key}")

            # Call the function passing the active client
            for attempt in range(1, max_attempts_per_key + 1):
                try:
                    return func(active_client)
                except Exception as e:
                    err_msg = str(e)
                    is_quota_error = (
                        "429" in err_msg or 
                        "resource_exhausted" in err_msg.lower() or 
                        "quota" in err_msg.lower()
                    )

                    if is_quota_error:
                        logger.warning(f"Gemini API quota exceeded on key {masked_key}. Rotating key...")
                        self.cooldowns[active_key] = time.time() + 60.0  # Cooldown for 60 seconds
                        tried_keys.add(active_key)
                        break  # Break inner loop to try next key
                    else:
                        is_retryable = "500" in err_msg or "503" in err_msg or "unavailable" in err_msg.lower()
                        if is_retryable and attempt < max_attempts_per_key:
                            wait_time = base_delay * (2 ** (attempt - 1))
                            logger.warning(f"Transient error on key {masked_key}: {e}. Retrying in {wait_time:.1f}s...")
                            time.sleep(wait_time)
                        else:
                            raise e

        raise RuntimeError("All configured Gemini API keys are exhausted or rate-limited.")

    def extract_structured_data(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: Type[BaseModel],
        pdf_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = 8192
    ) -> Tuple[str, int, int, float]:
        """Calls Gemini to extract structured data matching the JSON Schema, with auto-retry."""
        logger.info(f"Sending extraction request to Gemini ({model_name}, PDF: {pdf_path is not None})...")

        # Build content parts
        parts = []

        # If PDF path provided, upload PDF as inline data
        if pdf_path:
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
                logger.info(f"Loaded PDF '{pdf_path}' as inline bytes for multimodal input.")
            except Exception as e:
                logger.error(f"Failed to read PDF for multimodal input: {e}")

        # Add the text prompt
        parts.append(types.Part.from_text(text=prompt))

        # Build config
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        def _call(client):
            return client.models.generate_content(
                model=model_name,
                contents=[types.Content(role="user", parts=parts)],
                config=config
            )

        response = self.execute_with_retry_and_rotation(_call)

        # Extract text and usage
        raw_text = response.text or ""
        usage = response.usage_metadata
        prompt_tokens = (usage.prompt_token_count or 0) if usage else 0
        candidate_tokens = (usage.candidates_token_count or 0) if usage else 0
        cost = compute_gemini_cost(prompt_tokens, candidate_tokens, model_name)

        return raw_text, prompt_tokens, candidate_tokens, cost

    def mutate_prompt(
        self,
        system_instruction: str,
        user_prompt: str,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.7
    ) -> Tuple[str, int, int, float]:
        """Sends prompt mutation request to Gemini, with auto-retry."""
        logger.info(f"Sending prompt mutation request to Gemini ({model_name})...")

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=16384,
        )

        def _call(client):
            return client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config
            )

        response = self.execute_with_retry_and_rotation(_call)

        raw_text = response.text or ""
        usage = response.usage_metadata
        prompt_tokens = (usage.prompt_token_count or 0) if usage else 0
        candidate_tokens = (usage.candidates_token_count or 0) if usage else 0
        cost = compute_gemini_cost(prompt_tokens, candidate_tokens, model_name)

        return raw_text, prompt_tokens, candidate_tokens, cost

    def check_semantic_similarity(
        self,
        pred: str,
        gold: str,
        model_name: str = FLASH_MODEL
    ) -> bool:
        """Determines if prediction is semantically equivalent to gold reference using Gemini."""
        logger.info(f"Checking semantic similarity using Gemini ({model_name})...")

        system_instruction = (
            "You are a strict, objective evaluation assistant. "
            "Your task is to determine if a PREDICTED text extracted from a document is semantically equivalent to the GOLD reference text. "
            "Identify if they convey the same core information (dates, names, values, meaning), even if the phrasing, punctuation, or formatting differs slightly. "
            "Do not allow conflicting facts or missing key details. "
            "Respond with exactly one word: 'YES' if they are semantically equivalent, and 'NO' if they are not. "
            "Do not explain your reasoning."
        )

        user_prompt = f"GOLD: {gold}\nPREDICTED: {pred}"

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            max_output_tokens=5,
        )

        try:
            def _call(client):
                return client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config
                )

            response = self.execute_with_retry_and_rotation(_call)
            content = response.text or ""
            return "YES" in content.strip().upper()
        except Exception as e:
            logger.warning(f"Gemini semantic check failed: {e}")
            return False
