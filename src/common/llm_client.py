"""
LLM Client
Provider-agnostic wrapper for LLM API calls.

Lives in `src.common` rather than `src.generation` because it's shared
infrastructure, not a generation-only concern: ingestion (chunker.py,
metadata_extractor.py) and the retrieval layer's LLM-based reranker and
retrieval agent all call `simple_generate()` directly, alongside the
generation orchestrator's full `LLMClient`.

Handles:
- Provider routing (Gemini, OpenAI, Anthropic)
- Retry with exponential backoff
- Timeout enforcement
- Rate limit handling
- Usage stats extraction (provider-agnostic)
- Model info tracking

Architecture:
    Prompt → LLMClient.generate() → GeneratedAnswer

Configuration-driven — models from config/generation/models.yaml.
No direct SDK calls in business logic.
"""

import os
import re
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from dotenv import load_dotenv

from src.generation.config import (
    Prompt,
    GeneratedAnswer,
    UsageStats,
    ModelInfo,
    ModelConfig,
)

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# Provider Exceptions
# ============================================================================

class ProviderException(Exception):
    """Base exception for provider-related errors."""
    pass


class ProviderCredentialError(ProviderException):
    """Raised when provider credentials are missing or invalid."""
    pass


class ProviderAPIError(ProviderException):
    """Raised when the provider API returns an error."""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Any] = None,
        retry_after_seconds: Optional[float] = None,
        is_daily_quota: bool = False,
    ):
        self.status_code = status_code
        self.response = response
        # Best-effort — how long the provider says to wait before retrying
        # (parsed from structured RetryInfo or free-text, see
        # _extract_retry_after below). None when unknown.
        self.retry_after_seconds = retry_after_seconds
        # Whether this looks like a per-day quota (vs. a short per-minute
        # rate limit) — the frontend uses this to warn that the real wait
        # may be much longer than retry_after_seconds suggests.
        self.is_daily_quota = is_daily_quota
        super().__init__(message)


class ProviderRateLimitError(ProviderAPIError):
    """Raised when rate limit is exceeded."""
    pass


class ProviderTimeoutError(ProviderAPIError):
    """Raised when request times out."""
    pass


def _extract_retry_after(exc: Exception, error_msg: str) -> Tuple[Optional[float], bool]:
    """
    Best-effort extraction of "how long to wait before retrying" from a
    provider error, plus whether it looks like a per-day (not per-minute)
    quota.

    Tries the structured path first (Google API errors carry a RetryInfo
    proto in `.details` when the transport preserves gRPC trailer
    metadata), then falls back to regexing the free-text error message,
    since REST-transport errors don't always populate `.details`.

    Returns:
        (retry_after_seconds, is_daily_quota) — either/both may be
        None/False if not determinable.
    """
    retry_after: Optional[float] = None

    details = getattr(exc, "details", None)
    if callable(details):
        try:
            details = details()
        except Exception:
            details = None
    if details:
        for detail in details:
            retry_delay = getattr(detail, "retry_delay", None)
            if retry_delay is not None:
                try:
                    retry_after = retry_delay.seconds + retry_delay.nanos / 1e9
                    break
                except AttributeError:
                    pass

    if retry_after is None:
        match = re.search(r"(?:retry|try again) in ([\d.]+)\s*s", error_msg, re.IGNORECASE)
        if match:
            try:
                retry_after = float(match.group(1))
            except ValueError:
                pass

    if retry_after is None:
        match = re.search(r"seconds:\s*(\d+)", error_msg)
        if match:
            try:
                retry_after = float(match.group(1))
            except ValueError:
                pass

    is_daily_quota = "perday" in error_msg.lower().replace(" ", "").replace("-", "")

    return retry_after, is_daily_quota


# ============================================================================
# Provider Adapters
# ============================================================================

class ProviderAdapter(ABC):
    """
    Abstract adapter for LLM providers.
    
    Each provider (Gemini, OpenAI, Anthropic) implements this interface.
    The LLMClient delegates to the appropriate adapter based on config.
    
    Adapters RAISE exceptions for errors — they do NOT return error responses.
    Retry logic is handled by LLMClient.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        config: ModelConfig,
        timeout_seconds: int
    ) -> GeneratedAnswer:
        """
        Generate a response from the provider.
        
        Args:
            prompt: Structured Prompt object
            config: Model configuration
            timeout_seconds: Request timeout
            
        Returns:
            GeneratedAnswer with content and usage stats
            
        Raises:
            ProviderCredentialError: Missing/invalid credentials
            ProviderAPIError: API returned an error
            ProviderRateLimitError: Rate limit exceeded
            ProviderTimeoutError: Request timed out
        """
        ...
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Check if provider credentials are available."""
        ...


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google Gemini API."""
    
    def __init__(self):
        self._client = None
        self._model = None
    
    def _initialize(self, model_name: str):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ProviderCredentialError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ProviderCredentialError(
                    "GEMINI_API_KEY environment variable is required for Gemini models"
                )
            
            try:
                genai.configure(api_key=api_key)
                self._client = genai
                self._model = genai.GenerativeModel(model_name)
            except Exception as e:
                raise ProviderCredentialError(f"Failed to initialize Gemini client: {e}")
    
    def generate(
        self,
        prompt: Prompt,
        config: ModelConfig,
        timeout_seconds: int
    ) -> GeneratedAnswer:
        """Generate using Gemini API."""
        start_time = time.time()
        
        self._initialize(config.model_name)
        
        # Build generation config
        generation_config = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "top_p": config.top_p,
        }
        
        try:
            # Combine system and user prompts
            contents = []
            if prompt.system_prompt:
                contents.append({"role": "user", "parts": [prompt.system_prompt]})
                contents.append({"role": "model", "parts": ["Understood. I will follow these instructions."]})
            contents.append({"role": "user", "parts": [prompt.user_prompt]})
            
            response = self._model.generate_content(
                contents=contents,
                generation_config=generation_config,
                request_options={"timeout": timeout_seconds * 1000}  # Convert to ms
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            # Check for blocked/empty responses
            if not response.candidates:
                block_reason = getattr(response, 'prompt_feedback', {}).get('block_reason', 'unknown')
                raise ProviderAPIError(
                    f"Response blocked: {block_reason}",
                    response=response
                )
            
            # Extract usage stats
            usage = UsageStats(
                input_tokens=getattr(response.usage_metadata, 'prompt_token_count', 0),
                output_tokens=getattr(response.usage_metadata, 'candidates_token_count', 0),
                total_tokens=getattr(response.usage_metadata, 'total_token_count', 0),
                provider_metadata={
                    "model": config.model_name,
                    "finish_reason": getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'unknown',
                }
            )
            
            return GeneratedAnswer(
                content=response.text,
                model_info=config.to_model_info(),
                usage=usage,
                generation_time_ms=round(generation_time, 2),
                finish_reason=str(getattr(response.candidates[0], 'finish_reason', 'stop')) if response.candidates else 'stop',
                raw_response=response
            )
            
        except Exception as e:
            # Re-raise with appropriate provider error
            if isinstance(e, ProviderException):
                raise
            
            # Convert common Gemini errors
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                retry_after, is_daily = _extract_retry_after(e, error_msg)
                raise ProviderRateLimitError(
                    f"Gemini rate limit exceeded: {error_msg}",
                    response=e,
                    retry_after_seconds=retry_after,
                    is_daily_quota=is_daily,
                )
            elif "timeout" in error_msg.lower():
                raise ProviderTimeoutError(f"Gemini timeout: {error_msg}", response=e)
            else:
                raise ProviderAPIError(f"Gemini API error: {error_msg}", response=e)
    
    def validate_credentials(self) -> bool:
        """Check if Gemini API key is configured."""
        try:
            return bool(os.getenv("GEMINI_API_KEY"))
        except Exception:
            return False


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API."""
    
    def __init__(self):
        self._client = None
    
    def _initialize(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderCredentialError(
                    "openai package not installed. "
                    "Run: pip install openai"
                )
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ProviderCredentialError(
                    "OPENAI_API_KEY environment variable is required for OpenAI models"
                )
            
            try:
                self._client = OpenAI(api_key=api_key)
            except Exception as e:
                raise ProviderCredentialError(f"Failed to initialize OpenAI client: {e}")
    
    def generate(
        self,
        prompt: Prompt,
        config: ModelConfig,
        timeout_seconds: int
    ) -> GeneratedAnswer:
        """Generate using OpenAI API."""
        start_time = time.time()
        
        self._initialize()
        
        messages = []
        if prompt.system_prompt:
            messages.append({"role": "system", "content": prompt.system_prompt})
        messages.append({"role": "user", "content": prompt.user_prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_output_tokens,
                top_p=config.top_p,
                timeout=timeout_seconds
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            if not response.choices:
                raise ProviderAPIError("No choices returned from OpenAI", response=response)
            
            usage = UsageStats(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                provider_metadata={
                    "model": config.model_name,
                    "finish_reason": response.choices[0].finish_reason if response.choices else 'unknown',
                }
            )
            
            return GeneratedAnswer(
                content=response.choices[0].message.content if response.choices else "",
                model_info=config.to_model_info(),
                usage=usage,
                generation_time_ms=round(generation_time, 2),
                finish_reason=str(response.choices[0].finish_reason) if response.choices else 'stop',
                raw_response=response
            )
            
        except Exception as e:
            # Re-raise with appropriate provider error
            if isinstance(e, ProviderException):
                raise
            
            # Convert common OpenAI errors
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                retry_after, is_daily = _extract_retry_after(e, error_msg)
                raise ProviderRateLimitError(
                    f"OpenAI rate limit exceeded: {error_msg}",
                    response=e,
                    retry_after_seconds=retry_after,
                    is_daily_quota=is_daily,
                )
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                raise ProviderTimeoutError(f"OpenAI timeout: {error_msg}", response=e)
            else:
                raise ProviderAPIError(f"OpenAI API error: {error_msg}", response=e)
    
    def validate_credentials(self) -> bool:
        """Check if OpenAI API key is configured."""
        try:
            return bool(os.getenv("OPENAI_API_KEY"))
        except Exception:
            return False


class DeepSeekAdapter(OpenAIAdapter):
    """
    Adapter for DeepSeek's API.

    DeepSeek's API is OpenAI-compatible (same request/response shape), so
    this only overrides client initialization (different env var and
    base_url) and `generate()` (identical to OpenAIAdapter's, just with
    "DeepSeek" in error messages instead of "OpenAI" for clearer logs).
    """

    def _initialize(self):
        """Lazy initialization of the DeepSeek client (OpenAI SDK, custom base_url)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderCredentialError(
                    "openai package not installed. "
                    "Run: pip install openai"
                )

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ProviderCredentialError(
                    "DEEPSEEK_API_KEY environment variable is required for DeepSeek models"
                )

            try:
                self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            except Exception as e:
                raise ProviderCredentialError(f"Failed to initialize DeepSeek client: {e}")

    def generate(
        self,
        prompt: Prompt,
        config: ModelConfig,
        timeout_seconds: int
    ) -> GeneratedAnswer:
        """Generate using DeepSeek's OpenAI-compatible API."""
        start_time = time.time()

        self._initialize()

        messages = []
        if prompt.system_prompt:
            messages.append({"role": "system", "content": prompt.system_prompt})
        messages.append({"role": "user", "content": prompt.user_prompt})

        try:
            response = self._client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_output_tokens,
                top_p=config.top_p,
                timeout=timeout_seconds
            )

            generation_time = (time.time() - start_time) * 1000

            if not response.choices:
                raise ProviderAPIError("No choices returned from DeepSeek", response=response)

            usage = UsageStats(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                provider_metadata={
                    "model": config.model_name,
                    "finish_reason": response.choices[0].finish_reason if response.choices else 'unknown',
                }
            )

            return GeneratedAnswer(
                content=response.choices[0].message.content if response.choices else "",
                model_info=config.to_model_info(),
                usage=usage,
                generation_time_ms=round(generation_time, 2),
                finish_reason=str(response.choices[0].finish_reason) if response.choices else 'stop',
                raw_response=response
            )

        except Exception as e:
            if isinstance(e, ProviderException):
                raise

            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                retry_after, is_daily = _extract_retry_after(e, error_msg)
                raise ProviderRateLimitError(
                    f"DeepSeek rate limit exceeded: {error_msg}",
                    response=e,
                    retry_after_seconds=retry_after,
                    is_daily_quota=is_daily,
                )
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                raise ProviderTimeoutError(f"DeepSeek timeout: {error_msg}", response=e)
            else:
                raise ProviderAPIError(f"DeepSeek API error: {error_msg}", response=e)

    def validate_credentials(self) -> bool:
        """Check if DeepSeek API key is configured."""
        try:
            return bool(os.getenv("DEEPSEEK_API_KEY"))
        except Exception:
            return False


# ============================================================================
# Provider Registry
# ============================================================================

class ProviderRegistry:
    """
    Registry of available LLM providers.
    
    Maps provider names to adapter instances.
    New providers can be registered at runtime.
    """
    
    def __init__(self):
        self._adapters: Dict[str, ProviderAdapter] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register built-in providers."""
        self.register("gemini", GeminiAdapter())
        self.register("openai", OpenAIAdapter())
        self.register("deepseek", DeepSeekAdapter())
    
    def register(self, name: str, adapter: ProviderAdapter):
        """Register a provider adapter."""
        self._adapters[name.lower()] = adapter
    
    def get(self, provider_name: str) -> ProviderAdapter:
        """
        Get adapter for a provider.
        
        Args:
            provider_name: Provider name (case-insensitive)
            
        Returns:
            ProviderAdapter instance
            
        Raises:
            ValueError: If provider is not registered
        """
        adapter = self._adapters.get(provider_name.lower())
        if adapter is None:
            available = list(self._adapters.keys())
            raise ValueError(
                f"Unknown provider: '{provider_name}'. "
                f"Available: {available}"
            )
        return adapter
    
    def list_providers(self) -> List[str]:
        """List all registered providers."""
        return list(self._adapters.keys())


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """
    Provider-agnostic LLM client with retry, timeout, and rate limit handling.
    
    Usage:
        client = LLMClient()
        answer = client.generate(prompt, model_config)
    """
    
    def __init__(self, provider_registry: Optional[ProviderRegistry] = None):
        """
        Initialize the LLM client.
        
        Args:
            provider_registry: ProviderRegistry instance (creates default if None)
        """
        self.registry = provider_registry or ProviderRegistry()
    
    def generate(
        self,
        prompt: Prompt,
        config: ModelConfig
    ) -> GeneratedAnswer:
        """
        Generate a response from the configured LLM.
        
        Handles retry with exponential backoff, timeout, and rate limits.
        
        Args:
            prompt: Structured Prompt object
            config: Model configuration
            
        Returns:
            GeneratedAnswer with content and usage stats
        """
        adapter = self.registry.get(config.provider)

        # Validate credentials before attempting
        if not adapter.validate_credentials():
            return GeneratedAnswer(
                content="",
                model_info=config.to_model_info(),
                usage=UsageStats(),
                generation_time_ms=0,
                finish_reason="error",
                error_type="credential_error",
                raw_response=None
            )
        
        last_error = None
        
        for attempt in range(config.max_retries + 1):
            try:
                result = adapter.generate(
                    prompt=prompt,
                    config=config,
                    timeout_seconds=config.timeout_seconds
                )
                
                # Ensure we got content
                if not result.content or result.finish_reason in ("error", "unknown"):
                    # This shouldn't happen with our improved adapters, but keep as safeguard
                    raise ProviderAPIError(f"Empty or invalid response: {result.finish_reason}")
                
                return result
                
            except ProviderRateLimitError as e:
                last_error = e
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{config.max_retries + 1}): {e}"
                )
                
            except ProviderTimeoutError as e:
                last_error = e
                logger.warning(
                    f"Timeout (attempt {attempt + 1}/{config.max_retries + 1}): {e}"
                )
                
            except (ProviderAPIError, ProviderCredentialError) as e:
                # Don't retry credential errors; for API errors, retry unless they're fatal
                if isinstance(e, ProviderCredentialError):
                    logger.error(f"Credential error: {e}")
                    break
                
                last_error = e
                logger.warning(
                    f"API error (attempt {attempt + 1}/{config.max_retries + 1}): {e}"
                )
                
            except Exception as e:
                # Catch-all for unexpected errors
                last_error = e
                logger.warning(
                    f"Unexpected error (attempt {attempt + 1}/{config.max_retries + 1}): {e}"
                )
            
            # Don't sleep after the last attempt
            if attempt < config.max_retries:
                delay = config.retry_delay_seconds * (2 ** attempt)
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
        
        # All retries exhausted
        logger.error(f"All {config.max_retries + 1} attempts failed. Last error: {last_error}")

        if isinstance(last_error, ProviderRateLimitError):
            error_type = "rate_limited"
        elif isinstance(last_error, ProviderCredentialError):
            error_type = "credential_error"
        elif isinstance(last_error, ProviderTimeoutError):
            error_type = "timeout"
        elif last_error is not None:
            error_type = "api_error"
        else:
            error_type = None

        return GeneratedAnswer(
            content="",
            model_info=config.to_model_info(),
            usage=UsageStats(),
            generation_time_ms=0,
            finish_reason="error",
            error_type=error_type,
            retry_after_seconds=getattr(last_error, "retry_after_seconds", None),
            is_daily_quota=getattr(last_error, "is_daily_quota", False),
            raw_response=None
        )


# ============================================================================
# Simple Single-Turn Helper
# ============================================================================

def simple_generate(
    prompt: str,
    model_name: str = "deepseek-chat",
    temperature: float = 0.3,
    max_output_tokens: int = 2048,
    timeout_seconds: int = 30,
) -> str:
    """
    Minimal single-turn text completion for callers that just need a raw
    prompt-in/text-out call — document metadata extraction, semantic
    chunking, retrieval-agent reasoning, LLM reranking — without building a
    full GenerationRequest.

    Deliberately calls the adapter directly rather than going through
    LLMClient.generate()'s retry loop: every caller of this helper already
    has its own outer retry/fallback logic (e.g. extract_with_retry(),
    fallback to uniform chunking, fallback to DIRECT_RETRIEVE), so adding a
    second, inner retry loop here would just compound backoff delays.

    Raises ProviderRateLimitError/ProviderAPIError/ProviderCredentialError/
    ProviderTimeoutError on failure — same as any ProviderAdapter.
    """
    from src.generation.config import Prompt, ModelConfig

    adapter = DeepSeekAdapter()
    config = ModelConfig(
        provider="deepseek",
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    prompt_obj = Prompt(
        system_prompt="",
        user_prompt=prompt,
        template_id="simple",
        template_version="1.0.0",
        context_token_count=0,
        system_token_count=0,
        user_token_count=0,
        total_token_count=0,
        prompt_id="simple",
        request_id="simple",
    )
    return adapter.generate(prompt_obj, config, timeout_seconds=timeout_seconds).content


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM Client - Validation")
    print("=" * 60)
    
    # Load configuration
    from src.generation.config import GenerationConfig, GenerationMode, ModeConfig, Prompt
    
    gen_config = GenerationConfig.from_yaml()
    
    # Debug: Check if models loaded
    print(f"\n📋 Debug: models loaded = {len(gen_config.models)}")
    print(f"   Model keys: {list(gen_config.models.keys())}")
    
    print(f"\n📋 Available models:")
    if gen_config.models:
        for key, model in gen_config.models.items():
            print(f"   {key}: {model.provider}/{model.model_name} "
                  f"(temp={model.temperature}, max_tokens={model.max_output_tokens})")
    else:
        print("   ⚠️  No models loaded! Check config/generation/models.yaml")
        print("   Expected format:")
        print("   models:")
        print("     deepseek_chat:")
        print("       provider: deepseek")
        print("       model_name: deepseek-chat")
        print("       temperature: 0.7")
        print("       max_output_tokens: 1024")
        # Exit early if no models
        exit(1)
    
    # Create a single registry instance and reuse it
    registry = ProviderRegistry()
    
    print(f"\n📋 Registered providers: {registry.list_providers()}")
    
    # Check credentials using the same registry instance
    for provider_name in registry.list_providers():
        adapter = registry.get(provider_name)
        has_creds = adapter.validate_credentials()
        print(f"   {provider_name}: {'✅ credentials found' if has_creds else '❌ missing credentials'}")
    
    # Build a sample prompt
    prompt = Prompt(
        system_prompt="You are a helpful academic assistant.",
        user_prompt="## STUDY MATERIAL CONTEXT\n[SCM | Inventory | Slide 5]\nInventory refers to the stock of goods...\n\n## QUESTION\nWhat is inventory?",
        template_id="context_aware",
        template_version="1.0.0",
        context_token_count=50,
        system_token_count=15,
        user_token_count=80,
        total_token_count=95,
        prompt_id="test-prompt-001",
        request_id="test-001",
        mode=GenerationMode.CONTEXT_AWARE,
    )
    
    # Test with DeepSeek if credentials available - using the same registry
    deepseek_config = gen_config.models.get("deepseek_chat")

    # Check both config and credentials using the shared registry
    if deepseek_config and registry.get("deepseek").validate_credentials():
        print(f"\n{'─' * 50}")
        print("Testing DeepSeek generation...")

        client = LLMClient(provider_registry=registry)
        answer = client.generate(prompt, deepseek_config)
        
        print(f"   Model: {answer.model_info.provider}/{answer.model_info.model_name}")
        print(f"   Finish reason: {answer.finish_reason}")
        print(f"   Time: {answer.generation_time_ms:.0f}ms")
        print(f"   Tokens: {answer.usage.input_tokens} in / {answer.usage.output_tokens} out / {answer.usage.total_tokens} total")
        print(f"   Content preview: {answer.content[:200]}...")
        
        # Check if we got a valid response
        if answer.finish_reason == "error":
            print(f"\n   ⚠️  Generation failed - check logs above")
        else:
            print(f"\n   ✅ Generation successful!")
            
    else:
        # Clearer diagnostic message
        if not deepseek_config:
            print(f"\n⚠️  DeepSeek model 'deepseek_chat' not found in config")
            print(f"   Available models: {list(gen_config.models.keys())}")
        elif not registry.get("deepseek").validate_credentials():
            print(f"\n⚠️  DeepSeek credentials not found")
            print("   Set DEEPSEEK_API_KEY environment variable")
        else:
            print(f"\n⚠️  Unknown error - check configuration")
    
    print(f"\n✅ LLM Client ready for integration")