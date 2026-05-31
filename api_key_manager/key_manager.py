import os
import time
import logging
import threading
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError, APIError

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("APIKeyManager")


class APIKeyManager:
    """Thread-safe manager for loading, rotating, and tracking cooling down API keys."""

    def __init__(self, key_patterns: List[str] = ["KEY_1", "KEY_2", "KEY_3"], cooldown_duration: float = 60.0):
        self.cooldown_duration = cooldown_duration
        self.lock = threading.Lock()

        # Load environment variables
        load_dotenv()
        
        self.keys: List[str] = []
        
        # Load keys flexibly matching standard formats (KEY_1, key 2, 3)
        for pattern in key_patterns:
            norm_pattern = pattern.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
            found_val = None
            
            # Scan environment keys to find matching names (case-insensitive and format-flexible)
            for env_key, env_val in os.environ.items():
                norm_env_key = env_key.lower().replace(" ", "").replace("_", "").replace("-", "")
                if norm_env_key == norm_pattern or norm_env_key == f"key{norm_pattern}":
                    found_val = env_val
                    break
            
            # Directly lookup original pattern name if not found by normalized scan
            if not found_val:
                found_val = os.environ.get(pattern)
                
            if found_val:
                self.keys.append(found_val)

        # Remove duplicate keys and empty spaces
        self.keys = list(dict.fromkeys([k for k in self.keys if k and k.strip()]))

        if not self.keys:
            logger.error("No valid API keys could be loaded. Please ensure environment variables are configured correctly.")

        # Cooldown map: key -> timestamp when cooldown finishes
        self.cooldowns: Dict[str, float] = {}
        self.current_index = 0

    def mark_failed(self, key: str):
        """Puts an exhausted or rate-limited API key on cooldown."""
        with self.lock:
            cooldown_end = time.time() + self.cooldown_duration
            self.cooldowns[key] = cooldown_end
            masked_key = f"...{key[-6:]}" if len(key) > 6 else "key"
            logger.warning(f"API Key {masked_key} marked as exhausted. Cooldown set for {self.cooldown_duration}s.")

    def get_available_key(self) -> str:
        """Rotates keys and returns the next key that is not in cooldown."""
        with self.lock:
            if not self.keys:
                raise ValueError("No API keys loaded. Please check your environment variables.")

            num_keys = len(self.keys)
            now = time.time()

            # Iterate through the list of keys exactly once
            for _ in range(num_keys):
                candidate_key = self.keys[self.current_index]
                cooldown_until = self.cooldowns.get(candidate_key, 0.0)

                # Step index forward (round-robin rotation)
                self.current_index = (self.current_index + 1) % num_keys

                if now >= cooldown_until:
                    return candidate_key

            # Fallback: if all keys are currently in cooldown, select the one that cools down earliest
            logger.warning("All API keys are in cooldown. Reverting to key nearest to cooldown expiration.")
            min_cooldown_key = min(self.keys, key=lambda k: self.cooldowns.get(k, 0.0))
            return min_cooldown_key


class RotationalChatCompletions:
    """Namespace handler that intercepts 'chat.completions.create' to apply rotation and retries."""

    def __init__(self, key_manager: APIKeyManager, base_delay: float = 1.0, max_retries_per_key: int = 2):
        self.key_manager = key_manager
        self.base_delay = base_delay
        self.max_retries_per_key = max_retries_per_key

    def create(self, *args, **kwargs) -> Any:
        """Performs OpenAI chat creation with automatic key rotation and transient backoff retries."""
        tried_keys = set()
        total_keys = len(self.key_manager.keys)

        # Keep attempting until we've exhausted all available keys
        while len(tried_keys) < total_keys or not total_keys:
            active_key = self.key_manager.get_available_key()
            masked_key = f"...{active_key[-6:]}" if len(active_key) > 6 else "key"
            logger.info(f"Issuing request with API Key: {masked_key}")

            # Instantiate a client for the active key
            client = OpenAI(api_key=active_key)

            # Retry transient errors on the active key
            for attempt in range(1, self.max_retries_per_key + 1):
                try:
                    return client.chat.completions.create(*args, **kwargs)
                except RateLimitError as e:
                    # 429 Rate limit / Quota exceeded — mark key as failed and rotate
                    logger.error(f"RateLimitError (429) on active key: {e}. Shifting to next key.")
                    self.key_manager.mark_failed(active_key)
                    tried_keys.add(active_key)
                    break  # Break inner retry loop to request a new key
                except APIStatusError as e:
                    if e.status_code == 429:
                        logger.error(f"APIStatusError 429 (Rate Limit) on active key: {e}. Shifting to next key.")
                        self.key_manager.mark_failed(active_key)
                        tried_keys.add(active_key)
                        break  # Rotate
                    elif e.status_code in [500, 502, 503, 504]:
                        # Transient gateway or server error — retry with exponential backoff on same key
                        wait_time = self.base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"Transient Server Error ({e.status_code}) on attempt {attempt}/{self.max_retries_per_key}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        # Permanent API errors (e.g. 400 Bad Request, 401 Auth error) — raise immediately
                        logger.critical(f"Permanent API Error ({e.status_code}) encountered: {e}")
                        raise e
                except APIError as e:
                    # Connection or other structural API errors
                    logger.error(f"API Connection error on active key: {e}. Shifting to next key.")
                    self.key_manager.mark_failed(active_key)
                    tried_keys.add(active_key)
                    break  # Rotate
                except Exception as e:
                    logger.error(f"Unexpected structural execution error: {e}")
                    raise e

        raise RuntimeError("All configured API keys are exhausted or hit rate limits.")


class RotationalChatNamespace:
    """OpenAI-like client structure matching client.chat namespace."""
    
    def __init__(self, key_manager: APIKeyManager):
        self.completions = RotationalChatCompletions(key_manager)


class RotationalOpenAIClient:
    """Thread-safe OpenAI client wrapper supporting automatic key rotation on rate limits."""

    def __init__(self, key_patterns: List[str] = ["KEY_1", "KEY_2", "KEY_3"], cooldown_duration: float = 60.0):
        self.key_manager = APIKeyManager(key_patterns, cooldown_duration)
        self.chat = RotationalChatNamespace(self.key_manager)
