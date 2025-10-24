import os
import ssl
import langextract as lx
import dataclasses
import json
from typing import Any, Iterator, Sequence
import httpx

class GLMProviderSchema(lx.core.schema.BaseSchema):
    """Custom schema implementation for GLM provider using OpenAI-compatible API."""

    def __init__(self, schema_dict: dict[str, Any]):
        """Initialize the GLM schema.

        Args:
            schema_dict: The generated JSON schema dictionary.
        """
        self._schema_dict = schema_dict

    @property
    def schema_dict(self) -> dict[str, Any]:
        """Access the underlying schema dictionary.

        Returns:
            The JSON schema dictionary.
        """
        return self._schema_dict

    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[lx.data.ExampleData],
        attribute_suffix: str = "_attributes",
    ):
        """Generate schema from example data.

        Args:
            examples_data: Example extractions to learn from.
            attribute_suffix: Suffix for attribute fields.

        Returns:
            A configured GLMProviderSchema instance.
        """
        # Analyze examples to determine structure
        extraction_types = {}
        all_attributes = set()
        
        for example in examples_data:
            for extraction in example.extractions:
                class_name = extraction.extraction_class
                if class_name not in extraction_types:
                    extraction_types[class_name] = set()
                if extraction.attributes:
                    extraction_types[class_name].update(extraction.attributes.keys())
                    all_attributes.update(extraction.attributes.keys())

        # Build JSON schema based on extracted structure
        if extraction_types:
            schema_dict = {
                "type": "object",
                "properties": {
                    "extractions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "extraction_class": {
                                    "type": "string",
                                    "enum": list(extraction_types.keys())
                                },
                                "extraction_text": {"type": "string"},
                                "attributes": {
                                    "type": "object",
                                    "properties": {attr: {"type": "string"} for attr in sorted(all_attributes)},
                                    "additionalProperties": False
                                }
                            },
                            "required": ["extraction_class", "extraction_text"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["extractions"],
                "additionalProperties": False
            }
        else:
            # Fallback schema if no extractions found
            schema_dict = {
                "type": "object",
                "properties": {
                    "extractions": {
                        "type": "array", 
                        "items": {"type": "object"}
                    }
                }
            }

        return cls(schema_dict)

    def to_provider_config(self) -> dict[str, Any]:
        """Convert schema to provider-specific configuration.

        Returns:
            Dictionary of provider kwargs for the GLM model.
        """
        return {
            "response_schema": self._schema_dict,
            "structured_output": True
        }

    @property
    def supports_strict_mode(self) -> bool:
        """Whether this schema guarantees valid structured output.

        Returns:
            True if the provider will emit valid JSON without needing
            Markdown fences for extraction.
        """
        return True

@lx.providers.registry.register(r"^glm-", priority=10)
class GLMProviderLanguageModel(lx.core.base_model.BaseLanguageModel):
    """Custom LangExtract provider for GLM using OpenAI-compatible ChatCompletions API.

    This provider works with any GLM model that exposes an OpenAI-compatible
    ChatCompletions endpoint.

    Example usage:
        config = lx.factory.ModelConfig(
            model_id="glm-4",
            provider="GLMProvider",
            base_url="https://your-glm-api-endpoint.com/v1",
            api_key="your-api-key"
        )
        model = lx.factory.create_model(config)
    """

    model_id: str
    api_key: str | None
    base_url: str
    temperature: float
    verify_ssl: bool | str = False
    response_schema: dict[str, Any] | None = None
    structured_output: bool = False
    _client: httpx.AsyncClient | None = dataclasses.field(
        repr=False, compare=False, default=None
    )

    def __init__(
        self,
        model_id: str = "glm-4",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        verify_ssl: bool | str = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the GLM provider.

        Args:
            model_id: The GLM model ID.
            api_key: API key for the GLM service.
            base_url: Base URL for the GLM API endpoint.
            temperature: Sampling temperature.
            verify_ssl: Whether to verify SSL certificates. Can be True, False,
                       or path to a CA bundle.
            **kwargs: Additional parameters including schema configuration.
        """
        super().__init__()

        self.model_id = model_id
        self.api_key = api_key
        self.base_url = (
            base_url or "https://open.bigmodel.cn/api/paas/v4/"
        )  # Default GLM endpoint
        self.temperature = temperature
        self.verify_ssl = verify_ssl

        # Schema kwargs from GLMProviderSchema.to_provider_config()
        self.response_schema = kwargs.get("response_schema")
        self.structured_output = kwargs.get("structured_output", False)

        # Store any additional kwargs for potential use
        self._extra_kwargs = kwargs

        if not self.api_key:
            raise lx.exceptions.InferenceConfigError(
                "API key required. Set API_KEY environment variable or pass api_key parameter."
            )

    @classmethod
    def get_schema_class(cls) -> type[lx.core.schema.BaseSchema] | None:
        """Return our custom schema class.

        Returns:
            Our custom schema class that will be used to generate constraints.
        """
        return GLMProviderSchema

    def apply_schema(self, schema_instance: lx.core.schema.BaseSchema | None) -> None:
        """Apply or clear schema configuration.

        Args:
            schema_instance: The schema to apply, or None to clear existing schema.
        """
        super().apply_schema(schema_instance)

        if schema_instance:
            # Apply the new schema configuration
            config = schema_instance.to_provider_config()
            self.response_schema = config.get("response_schema")
            self.structured_output = config.get("structured_output", False)
        else:
            # Clear the schema configuration
            self.response_schema = None
            self.structured_output = False

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure the HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
                verify=self.verify_ssl,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[lx.core.types.ScoredOutput]]:
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: Input prompts to process.
            **kwargs: Additional generation parameters.

        Yields:
            Lists of ScoredOutputs, one per prompt.

        Raises:
            lx.exceptions.InferenceRuntimeError: If API call fails.
        """
        import asyncio

        # Run async inference synchronously for compatibility
        async def run_async_inference():
            results = []
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
                verify=self.verify_ssl,
            ) as client:
                for prompt in batch_prompts:
                    try:
                        result = await self._execute_single(client, prompt, kwargs)
                        results.append(result)
                    except Exception as e:
                        # Convert to LangExtract exception
                        raise lx.exceptions.InferenceRuntimeError(
                            f"GLM API error: {str(e)}", original=e
                        ) from e
            return results

        try:
            results = asyncio.run(run_async_inference())
            for result in results:
                yield result
        except Exception as e:
            if isinstance(e, lx.exceptions.InferenceRuntimeError):
                raise
            raise lx.exceptions.InferenceRuntimeError(
                f"GLM inference error: {str(e)}", original=e
            ) from e

    async def _execute_single(
        self, client: httpx.AsyncClient, prompt: str, kwargs: dict[str, Any]
    ) -> Sequence[lx.core.types.ScoredOutput]:
        """Execute a single inference request."""

        # Prepare the request payload for OpenAI-compatible API
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add optional parameters
        for key in ["max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        # Apply schema constraints if configured - use OpenAI-compatible format
        if self.structured_output:
            payload["response_format"] = {"type": "json_object"}

        # Make the API request
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        # Parse response
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]

        return [lx.core.types.ScoredOutput(score=1.0, output=content)]

    async def infer_async(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Sequence[Sequence[lx.core.types.ScoredOutput]]:
        """Async implementation of inference.

        Args:
            batch_prompts: Input prompts to process.
            **kwargs: Additional generation parameters.

        Returns:
            Lists of ScoredOutputs for all prompts.
        """
        client = self._ensure_client()
        results = []

        for prompt in batch_prompts:
            try:
                result = await self._execute_single(client, prompt, kwargs)
                results.append(result)
            except Exception as e:
                raise lx.exceptions.InferenceRuntimeError(
                    f"GLM API error: {str(e)}", original=e
                ) from e

        return results
