"""
title: Time Token And Cost Tracker
description: This function is designed to manage and calculate the costs associated with user interactions and model usage in a Open WebUI based on OpenRouters pricings. See https://github.com/bgeneto/open-webui-cost-tracker & https://openwebui.com/f/makimaki/cost_tracker/edit
author: Roni Laukkarinen (original code by Kkoolldd, maki, bgeneto)
author_url: https://github.com/ronilaukkarinen/open-webui-cost-tracker
funding_url: https://github.com/ronilaukkarinen/open-webui-cost-tracker
version: 1.2.16
license: MIT
requirements: requests, tiktoken, cachetools, pydantic
environment_variables:
disclaimer: This function is provided as is without any guarantees.
            It is your responsibility to ensure that the function meets your requirements.
            All metrics and costs are approximate and may vary depending on the model and the usage.
"""

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from threading import Lock
from typing import Any, Awaitable, Callable, Optional

import requests
import tiktoken
from cachetools import TTLCache, cached
from open_webui.utils.misc import get_last_assistant_message, get_messages_content
from pydantic import BaseModel, Field


class Config:
    DATA_DIR = "data"
    CACHE_DIR = os.path.join(DATA_DIR, ".cache")
    USER_COST_FILE = os.path.join(DATA_DIR, f"costs-{datetime.now().year}.json")
    CACHE_TTL = 432000  # try to keep model pricing json file for 5 days in the cache.
    CACHE_MAXSIZE = 16
    DECIMALS = "0.00000001"
    DEBUG_PREFIX = "DEBUG:    " + __name__ + " -"
    INFO_PREFIX = "INFO:     " + __name__ + " -"
    DEBUG = False


# Initialize cache
cache = TTLCache(maxsize=Config.CACHE_MAXSIZE, ttl=Config.CACHE_TTL)


def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        if Config.DEBUG:
            print(
                f"{Config.DEBUG_PREFIX} Encoding for model {model} not found. Using cl100k_base for computing tokens."
            )
        return tiktoken.get_encoding("cl100k_base")


class UserCostManager:
    def __init__(self, cost_file_path):
        self.cost_file_path = cost_file_path
        self._ensure_cost_file_exists()

    def _ensure_cost_file_exists(self):
        if not os.path.exists(self.cost_file_path):
            with open(self.cost_file_path, "w", encoding="UTF-8") as cost_file:
                json.dump({}, cost_file)

    def _read_costs(self):
        with open(self.cost_file_path, "r", encoding="UTF-8") as cost_file:
            return json.load(cost_file)

    def _write_costs(self, costs):
        with open(self.cost_file_path, "w", encoding="UTF-8") as cost_file:
            json.dump(costs, cost_file, indent=4)

    def update_user_cost(
        self,
        user_email: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_cost: Decimal,
    ):
        costs = self._read_costs()
        timestamp = datetime.now().isoformat()

        if user_email not in costs:
            costs[user_email] = []

        costs[user_email].append(
            {
                "model": model,
                "timestamp": timestamp,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": str(total_cost),
            }
        )

        self._write_costs(costs)


class ModelCostManager:
    _best_match_cache = {}

    def __init__(self, cache_dir=Config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.lock = Lock()
        self.url = (
            "https://openrouter.ai/api/v1/models"  # Changed URL to OpenRouter API
        )
        self.cache_file_path = self._get_cache_filename()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_filename(self):
        cache_file_name = hashlib.sha256(self.url.encode()).hexdigest() + ".json"
        return os.path.normpath(os.path.join(self.cache_dir, cache_file_name))

    def _is_cache_valid(self, cache_file_path):
        cache_file_mtime = os.path.getmtime(cache_file_path)
        return time.time() - cache_file_mtime < cache.ttl

    @cached(cache=cache)
    def get_cost_data(self):
        """
        Fetches model pricing data from OpenRouter API and transforms it into the required format.
        """
        with self.lock:
            if os.path.exists(self.cache_file_path) and self._is_cache_valid(
                self.cache_file_path
            ):
                with open(self.cache_file_path, "r", encoding="UTF-8") as cache_file:
                    if Config.DEBUG:
                        print(
                            f"{Config.DEBUG_PREFIX} Reading costs json file from disk!"
                        )
                    return json.load(cache_file)
        try:
            if Config.DEBUG:
                print(f"{Config.DEBUG_PREFIX} Downloading model costs json file!")
            response = requests.get(self.url)
            response.raise_for_status()
            raw_data = response.json()

            # Transform OpenRouter data format to match expected format
            transformed_data = {}
            for model in raw_data["data"]:
                model_id = model[
                    "id"
                ].lower()  # Ensure lowercase for consistent matching
                transformed_data[model_id] = {
                    "input_cost_per_token": float(model["pricing"]["prompt"]),
                    "output_cost_per_token": float(model["pricing"]["completion"]),
                    "context_window": model["context_length"],
                }
                if Config.DEBUG:
                    print(f"{Config.DEBUG_PREFIX} Added pricing for model: {model_id}")

            # backup existing cache file
            try:
                if os.path.exists(self.cache_file_path):
                    os.rename(self.cache_file_path, self.cache_file_path + ".bkp")
            except Exception as backup_error:
                print(f"**ERROR: Failed to backup costs json file. Error: {backup_error}")

            with self.lock:
                with open(self.cache_file_path, "w", encoding="UTF-8") as cache_file:
                    if Config.DEBUG:
                        print(f"{Config.DEBUG_PREFIX} Writing costs to json file!")
                    json.dump(transformed_data, cache_file, indent=2)
            return transformed_data

        except Exception as cost_data_error:
            print(
                f"**ERROR: Failed to download or write to costs json file. Using old cached file if available. Error: {cost_data_error}"
            )
            with self.lock:
                if os.path.exists(self.cache_file_path + ".bkp"):
                    with open(
                        self.cache_file_path + ".bkp", "r", encoding="UTF-8"
                    ) as cache_file:
                        if Config.DEBUG:
                            print(
                                f"{Config.DEBUG_PREFIX} Reading costs json file from backup!"
                            )
                        return json.load(cache_file)
                else:
                    raise cost_data_error

    def levenshtein_distance(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost
                )

        return dp[m][n]

    def _find_best_match(self, query: str, json_data) -> str:
        query_lower = query.lower()
        keys_lower = {key.lower(): key for key in json_data.keys()}

        # 1. 完全一致を確認
        if query_lower in keys_lower:
            return keys_lower[query_lower]

        # 2. 後方一致を確認
        for key_lower, original_key in keys_lower.items():
            if query_lower.endswith(key_lower):
                if Config.DEBUG:
                    print(
                        f"{Config.DEBUG_PREFIX} Found suffix match: {original_key} for query {query}"
                    )
                return original_key

        # 3. Levenshtein距離によるフォールバック
        threshold_ratio = 0.6 if len(query) < 15 else 0.3
        min_distance = float("inf")
        best_match = None
        threshold = round(len(query) * threshold_ratio)

        start = time.time()
        distances = (self.levenshtein_distance(query_lower, key) for key in keys_lower)
        for key, dist in zip(keys_lower.values(), distances):
            if dist < min_distance:
                min_distance = dist
                best_match = key
            if dist < 2:  # ほぼ完全一致の場合の早期終了
                return key
        end = time.time()

        if Config.DEBUG:
            print(
                f"{Config.DEBUG_PREFIX} Levenshtein distance search took {end - start:.3f} seconds"
            )

        if min_distance > threshold:
            return None  # しきい値内に一致するものが見つからなかった

        return best_match

    def _sanitize_model_name(self, name: str) -> str:
        """
        Keep the full model identifier for OpenRouter API matching
        Args:
            name (str): model name
        Returns:
            str: sanitized model name
        """
        # Remove only specific suffixes but keep the provider prefix
        suffixes = ["-tuned"]
        name = name.lower().strip()
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    def get_openai_pricing(self, model: str) -> dict:
        """
        Fallback OpenAI pricing for models not found in OpenRouter
        Uses official OpenAI pricing as of January 2025
        """
        model_lower = model.lower()

        # OpenAI GPT-4o series
        if "gpt-4o" in model_lower:
            if "mini" in model_lower:
                return {
                    "input_cost_per_token": 0.00000015,  # $0.15/1M tokens
                    "output_cost_per_token": 0.0000006,  # $0.60/1M tokens
                    "context_window": 128000,
                }
            else:
                return {
                    "input_cost_per_token": 0.0000025,   # $2.50/1M tokens (latest pricing)
                    "output_cost_per_token": 0.00001,    # $10.00/1M tokens
                    "context_window": 128000,
                }

        # OpenAI GPT-4.1 series (newest models)
        if "gpt-4.1" in model_lower:
            if "nano" in model_lower:
                return {
                    "input_cost_per_token": 0.0000001,   # $0.10/1M tokens
                    "output_cost_per_token": 0.0000004,  # $0.40/1M tokens
                    "context_window": 1047576,
                }
            elif "mini" in model_lower:
                return {
                    "input_cost_per_token": 0.0000004,   # $0.40/1M tokens
                    "output_cost_per_token": 0.0000016,  # $1.60/1M tokens
                    "context_window": 1047576,
                }
            else:
                return {
                    "input_cost_per_token": 0.000002,    # $2.00/1M tokens
                    "output_cost_per_token": 0.000008,   # $8.00/1M tokens
                    "context_window": 1047576,
                }

        # OpenAI GPT-4 series
        if "gpt-4" in model_lower and "turbo" in model_lower:
            return {
                "input_cost_per_token": 0.00001,        # $10.00/1M tokens
                "output_cost_per_token": 0.00003,       # $30.00/1M tokens
                "context_window": 128000,
            }
        elif "gpt-4" in model_lower:
            if "32k" in model_lower:
                return {
                    "input_cost_per_token": 0.00006,    # $60.00/1M tokens
                    "output_cost_per_token": 0.00012,   # $120.00/1M tokens
                    "context_window": 32000,
                }
            else:
                return {
                    "input_cost_per_token": 0.00003,    # $30.00/1M tokens
                    "output_cost_per_token": 0.00006,   # $60.00/1M tokens
                    "context_window": 8000,
                }

        # OpenAI o1 series (reasoning models)
        if "o1-preview" in model_lower or "o1-2024" in model_lower:
            return {
                "input_cost_per_token": 0.000015,       # $15.00/1M tokens
                "output_cost_per_token": 0.00006,       # $60.00/1M tokens
                "context_window": 128000,
            }
        elif "o1-mini" in model_lower:
            return {
                "input_cost_per_token": 0.000003,       # $3.00/1M tokens
                "output_cost_per_token": 0.000012,      # $12.00/1M tokens
                "context_window": 128000,
            }
        elif "o3-mini" in model_lower or "o4-mini" in model_lower:
            return {
                "input_cost_per_token": 0.0000011,      # $1.10/1M tokens
                "output_cost_per_token": 0.0000044,     # $4.40/1M tokens
                "context_window": 128000,
            }
        elif "o3" in model_lower and "2025" in model_lower:
            return {
                "input_cost_per_token": 0.00001,        # $10.00/1M tokens
                "output_cost_per_token": 0.00004,       # $40.00/1M tokens
                "context_window": 128000,
            }

        # OpenAI GPT-3.5 series
        if "gpt-3.5-turbo" in model_lower:
            if "instruct" in model_lower:
                return {
                    "input_cost_per_token": 0.0000015,  # $1.50/1M tokens
                    "output_cost_per_token": 0.000002,  # $2.00/1M tokens
                    "context_window": 4000,
                }
            elif "16k" in model_lower:
                return {
                    "input_cost_per_token": 0.000003,   # $3.00/1M tokens
                    "output_cost_per_token": 0.000004,  # $4.00/1M tokens
                    "context_window": 16000,
                }
            else:
                return {
                    "input_cost_per_token": 0.0000005,  # $0.50/1M tokens (latest)
                    "output_cost_per_token": 0.0000015, # $1.50/1M tokens
                    "context_window": 16000,
                }

        # OpenAI embedding models
        if "text-embedding-3-large" in model_lower:
            return {
                "input_cost_per_token": 0.00000013,     # $0.13/1M tokens
                "output_cost_per_token": 0,              # No output cost for embeddings
                "context_window": 8191,
            }
        elif "text-embedding-3-small" in model_lower:
            return {
                "input_cost_per_token": 0.00000002,     # $0.02/1M tokens
                "output_cost_per_token": 0,              # No output cost for embeddings
                "context_window": 8191,
            }
        elif "text-embedding-ada-002" in model_lower:
            return {
                "input_cost_per_token": 0.0000001,      # $0.10/1M tokens
                "output_cost_per_token": 0,              # No output cost for embeddings
                "context_window": 8191,
            }

        if Config.DEBUG:
            print(f"{Config.DEBUG_PREFIX} No OpenAI pricing found for model: {model}")
        return {}

    def get_model_data(self, model):
        if not model:
            return {}

        model = self._sanitize_model_name(model)
        json_data = self.get_cost_data()

        # Direct match first
        if model in json_data:
            return json_data[model]

        # Try finding in cache
        if model in ModelCostManager._best_match_cache:
            if Config.DEBUG:
                print(
                    f"{Config.DEBUG_PREFIX} Using cached costs for model named '{model}'"
                )
            cached_match = ModelCostManager._best_match_cache[model]
            return json_data.get(cached_match, {})

        # Find best match
        if Config.DEBUG:
            print(
                f"{Config.DEBUG_PREFIX} Searching best match in costs file for model named '{model}'"
            )

        # Look for partial matches in model IDs
        for model_id in json_data.keys():
            if model in model_id or model_id in model:
                ModelCostManager._best_match_cache[model] = model_id
                return json_data[model_id]

        # Try OpenAI pricing fallback for OpenAI models
        if ("gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower() or
            "o4" in model.lower() or "text-embedding" in model.lower() or
            "openai/" in model.lower()):
            openai_pricing = self.get_openai_pricing(model)
            if openai_pricing:
                if Config.DEBUG:
                    print(f"{Config.DEBUG_PREFIX} Using OpenAI fallback pricing for: {model}")
                return openai_pricing

        print(f"{Config.INFO_PREFIX} Model '{model}' not found in costs json file!")
        return {}


class CostCalculator:
    def __init__(
        self, user_cost_manager: UserCostManager, model_cost_manager: ModelCostManager
    ):
        self.model_cost_manager = model_cost_manager
        self.user_cost_manager = user_cost_manager

    def calculate_costs(
        self, model: str, input_tokens: int, output_tokens: int, compensation: float
    ) -> Decimal:
        model_pricing_data = self.model_cost_manager.get_model_data(model)
        if not model_pricing_data:
            print(f"{Config.INFO_PREFIX} Model '{model}' not found in costs json file!")
        input_cost_per_token = Decimal(
            str(model_pricing_data.get("input_cost_per_token", 0))
        )
        output_cost_per_token = Decimal(
            str(model_pricing_data.get("output_cost_per_token", 0))
        )

        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        total_cost = Decimal(float(compensation)) * (input_cost + output_cost)
        total_cost = total_cost.quantize(
            Decimal(Config.DECIMALS), rounding=ROUND_HALF_UP
        )

        return total_cost


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=15, description="Priority level")
        compensation: float = Field(
            default=1.0, description="Compensation for price calculation (percent)"
        )
        # New natural language format (enabled by default)
        use_natural_format: bool = Field(
            default=True,
            description="Use natural language format like '8 seconds, 3395 tokens and 0 € used'",
        )
        hide_zero_cost: bool = Field(
            default=True,
            description="Hide cost display when cost is zero (show only time and tokens)",
        )
        skip_free_models: bool = Field(
            default=True,
            description="Skip cost tracking for local/free models (models without OpenRouter pricing)",
        )
        disable_streaming_large_requests: bool = Field(
            default=True,
            description="Disable streaming for requests with >10k tokens to ensure cost tracking works properly",
        )
        large_request_token_threshold: int = Field(
            default=10000,
            description="Token threshold above which streaming will be disabled (if disable_streaming_large_requests is enabled)",
        )
        # Old format options (disabled by default)
        elapsed_time: bool = Field(
            default=False, description="Display the elapsed time (old format)"
        )
        number_of_tokens: bool = Field(
            default=False, description="Display total number of tokens (old format)"
        )
        tokens_per_sec: bool = Field(
            default=False, description="Display tokens per second metric (old format)"
        )
        debug: bool = Field(default=False, description="Display debugging messages")
        pass

    def __init__(self):
        self.valves = self.Valves()
        Config.DEBUG = self.valves.debug
        self.model_cost_manager = ModelCostManager()
        self.user_cost_manager = UserCostManager(Config.USER_COST_FILE)
        self.cost_calculator = CostCalculator(
            self.user_cost_manager, self.model_cost_manager
        )
        self.start_time = None
        self.input_tokens = 0
        self.processing_task = None
        self.processing_shown = False
        pass

    def _sanitize_model_name(self, name: str) -> str:
        """Sanitize model name while preserving provider prefix
        Args:
            name (str): model name
        Returns:
            str: sanitized model name
        """
        suffixes = ["-tuned"]
        name = name.lower().strip()
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    def _remove_roles(self, content):
        # Define the roles to be removed
        roles = ["SYSTEM:", "USER:", "ASSISTANT:", "PROMPT:"]

        # Process each line
        def process_line(line):
            for role in roles:
                if line.startswith(role):
                    return line.split(":", 1)[1].strip()
            return line  # Return the line unchanged if no role matches

        return "\n".join([process_line(line) for line in content.split("\n")])

    def _get_model(self, body):
        if "model" in body:
            return self._sanitize_model_name(body["model"])
        return None

    def _is_local_model(self, model: str, body: Optional[dict] = None) -> bool:
        """
        Detect if a model is local/free by checking model name patterns and known local indicators
        Args:
            model (str): model name
            body (dict): request body which may contain model metadata
        Returns:
            bool: True if model is detected as local/free
        """
        if not model:
            return True

        model_lower = model.lower()

        # Special handling for Auto Router - always attempt cost tracking
        if "auto" in model_lower or "router" in model_lower:
            if Config.DEBUG:
                print(f"{Config.DEBUG_PREFIX} Auto Router detected: {model} - will attempt cost tracking")
            return False  # Don't treat Auto Router as local, attempt to track costs

        # Check for common local model indicators in the model name
        local_model_indicators = [
            # Ollama models (typically just model name without provider prefix)
            "llama",
            "mistral",
            "qwen",
            "gemma",
            "phi",
            "codellama",
            "vicuna",
            "alpaca",
            "wizard",
            "orca",
            "nous",
            "dolphin",
            "neural",
            "chat",
            # Local model naming patterns
            ":latest",
            ":8b",
            ":7b",
            ":13b",
            ":70b",
            ":32b",
            ":1b",
            ":3b",
            # Local inference engines
            "ollama",
            "localai",
            "llamacpp",
            "koboldcpp",
            "textgen",
            "oobabooga",
        ]

        # If model name contains any local indicators, it's likely local
        if any(indicator in model_lower for indicator in local_model_indicators):
            if Config.DEBUG:
                print(
                    f"{Config.DEBUG_PREFIX} Local model indicator detected in name: {model}"
                )
            return True

        # Check if the model name lacks a provider prefix (OpenRouter models usually have prefixes)
        # External/paid models typically have formats like:
        # - "openai/gpt-4", "anthropic/claude-3", "google/gemini-pro"
        # - "meta-llama/llama-3.1-405b-instruct"
        # Local models typically just have the model name like:
        # - "llama3.1", "mistral", "qwen2:7b"

        # If it contains a slash, it's likely an external model with a provider
        if "/" in model:
            # But check if it's still a local setup (some local setups use provider/model format)
            provider = model.split("/")[0].lower()
            local_providers = ["local", "ollama", "llamacpp", "localai"]
            if provider in local_providers:
                if Config.DEBUG:
                    print(f"{Config.DEBUG_PREFIX} Local provider detected: {model}")
                return True

            # External provider detected (including OpenAI, Anthropic, Google, etc.)
            if Config.DEBUG:
                print(f"{Config.DEBUG_PREFIX} External provider detected: {model}")
            return False

        # Check for direct OpenAI model names (without slash prefix)
        openai_model_patterns = [
            "gpt-4", "gpt-3.5", "gpt-35", "o1-", "o3-", "o4-",
            "text-embedding", "davinci", "curie", "babbage", "ada"
        ]
        if any(pattern in model_lower for pattern in openai_model_patterns):
            if Config.DEBUG:
                print(f"{Config.DEBUG_PREFIX} OpenAI model detected (without prefix): {model}")
            return False

        # Simple model names without slashes are typically local Ollama models
        if Config.DEBUG:
            print(
                f"{Config.DEBUG_PREFIX} Simple model name detected (likely local): {model}"
            )
        return True

    def _get_auto_router_model(self, body: dict) -> str:
        """
        Try to extract the actual model used by Auto Router from response data
        Auto Router may provide the selected model in various places:
        - Response headers (x-model-used, model-used, etc.)
        - Response metadata
        - In the response body somewhere
        """
        model = self._get_model(body)

        if not model:
            return model

        model_lower = model.lower()
        if not ("auto" in model_lower or "router" in model_lower):
            return model

        if Config.DEBUG:
            print(f"{Config.DEBUG_PREFIX} Attempting to resolve Auto Router model: {model}")

        # Check if there's any model information in the response body
        # Auto Router might include the selected model in various fields
        potential_model_fields = [
            "selected_model", "model_used", "actual_model", "routed_model",
            "chosen_model", "target_model", "final_model"
        ]

        for field in potential_model_fields:
            if field in body and body[field]:
                detected_model = str(body[field])
                if Config.DEBUG:
                    print(f"{Config.DEBUG_PREFIX} Found Auto Router model in {field}: {detected_model}")
                return detected_model

        # Check in messages array for any model metadata
        if "messages" in body and isinstance(body["messages"], list):
            for message in body["messages"]:
                if isinstance(message, dict):
                    # Check if Auto Router added metadata to any message
                    for field in potential_model_fields + ["metadata", "model_info"]:
                        if field in message and message[field]:
                            if isinstance(message[field], dict):
                                # Look inside metadata objects
                                for subfield in potential_model_fields:
                                    if subfield in message[field]:
                                        detected_model = str(message[field][subfield])
                                        if Config.DEBUG:
                                            print(f"{Config.DEBUG_PREFIX} Found Auto Router model in message.{field}.{subfield}: {detected_model}")
                                        return detected_model
                            else:
                                detected_model = str(message[field])
                                if Config.DEBUG:
                                    print(f"{Config.DEBUG_PREFIX} Found Auto Router model in message.{field}: {detected_model}")
                                return detected_model

        # If we can't detect the specific model, we'll use a fallback
        # Use mid-tier model like GPT-3.5 for cost estimation
        fallback_models = [
            "openai/gpt-3.5-turbo",  # Most common fallback
            "anthropic/claude-3-haiku",  # Fast model
            "google/gemini-pro"  # Alternative
        ]

        for fallback in fallback_models:
            model_data = self.model_cost_manager.get_model_data(fallback)
            if model_data:
                if Config.DEBUG:
                    print(f"{Config.DEBUG_PREFIX} Auto Router fallback to {fallback} for cost estimation")
                return fallback

        if Config.DEBUG:
            print(f"{Config.DEBUG_PREFIX} No fallback model found for Auto Router, returning original")
        return model

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __user__: Optional[dict] = None,
    ) -> dict:

        Config.DEBUG = self.valves.debug

        try:
            # Get ALL message content including system prompts, user messages, etc.
            all_content = []
            for message in body.get("messages", []):
                if isinstance(message, dict) and "content" in message:
                    content = str(message["content"]).strip()
                    if content:  # Only add non-empty content
                        all_content.append(content)

            # Join all message content
            raw_content = "\n".join(all_content)
            if Config.DEBUG:
                print(
                    f"{Config.DEBUG_PREFIX} Raw content from {len(all_content)} messages: {len(raw_content)} chars"
                )
                print(
                    f"{Config.DEBUG_PREFIX} Raw content preview: {raw_content[:200]}..."
                )

            input_content = self._remove_roles(raw_content).strip()
            if Config.DEBUG:
                print(
                    f"{Config.DEBUG_PREFIX} After role removal: {len(input_content)} chars"
                )
                print(
                    f"{Config.DEBUG_PREFIX} Content preview: {input_content[:200]}..."
                )

            # Limit content size to prevent hanging (5MB max)
            max_content_size = 5 * 1024 * 1024  # 5MB
            if len(input_content) > max_content_size:
                if Config.DEBUG:
                    print(
                        f"{Config.DEBUG_PREFIX} Content too large ({len(input_content)} chars), truncating for token count"
                    )
                input_content = input_content[:max_content_size]

            # Fast token counting without tiktoken (prevents UI hanging)
            word_count = len(input_content.split())
            self.input_tokens = int(word_count * 1.3)
            if Config.DEBUG:
                print(
                    f"{Config.DEBUG_PREFIX} Fast token count: {word_count} words -> {self.input_tokens} tokens"
                )

        except Exception as input_error:
            if Config.DEBUG:
                print(f"{Config.DEBUG_PREFIX} Error processing input: {input_error}")
            # Fallback to zero tokens if everything fails
            self.input_tokens = 0

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Processing {self.input_tokens} input tokens...",
                    "done": False,
                },
            }
        )

        # Store a flag so we can clear the processing message in outlet
        self.processing_shown = True

        # Disable streaming for large requests to ensure cost tracking works properly
        if (self.valves.disable_streaming_large_requests and
            self.input_tokens > self.valves.large_request_token_threshold):
            if Config.DEBUG:
                print(
                    f"{Config.DEBUG_PREFIX} Disabling streaming for large request: {self.input_tokens} tokens > {self.valves.large_request_token_threshold} threshold"
                )
            body["stream"] = False

        # add user email to payload in order to track costs
        if __user__:
            if "email" in __user__:
                if Config.DEBUG:
                    print(
                        f"{Config.DEBUG_PREFIX} Adding email to request body: {__user__['email']}"
                    )
                body["user"] = __user__["email"]

        self.start_time = time.time()

        return body

    def stream(self, event: dict) -> dict:
        """
        Stream hook - called when AI starts streaming response chunks
        This is where we detect actual streaming start and hide processing message
        """
        # Mark that streaming has started so we know processing is done
        self.processing_shown = False

        # Just pass through the stream event unchanged
        return event

    def _format_euros(self, cost_eur):
        """Format cost in euros with smart rounding"""
        if cost_eur == 0:
            return "0 €"
        elif cost_eur < 0.0001:
            return "0 €"  # Treat very tiny costs as free
        elif cost_eur >= 0.98 and cost_eur <= 1.02:
            return "1 €"  # Round near 1 euro
        elif cost_eur >= 1.98 and cost_eur <= 2.02:
            return "2 €"  # Round near 2 euros
        elif cost_eur >= 2.98 and cost_eur <= 3.02:
            return "3 €"  # Round near 3 euros
        elif cost_eur >= 3.98 and cost_eur <= 4.02:
            return "4 €"  # Round near 4 euros
        elif cost_eur >= 4.98 and cost_eur <= 5.02:
            return "5 €"  # Round near 5 euros
        elif cost_eur >= 1:
            return f"{cost_eur:.0f} €"  # Round whole euros for larger amounts
        else:
            # For small costs, show meaningful decimals
            if cost_eur >= 0.01:
                return f"{cost_eur:.3f} €"
            else:
                return f"{cost_eur:.6f} €"

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:

        try:
            # Always clear any processing message at the start of outlet
            # This is critical for non-streaming APIs (OpenRouter, piped models) where stream() is never called
            if hasattr(self, 'processing_shown') and self.processing_shown:
                try:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Processing {self.input_tokens} input tokens...",
                                "done": True,
                            },
                        }
                    )
                    if Config.DEBUG:
                        print(f"{Config.DEBUG_PREFIX} Cleared stuck processing message for {self.input_tokens} tokens")
                    self.processing_shown = False
                except Exception as clear_error:
                    if Config.DEBUG:
                        print(f"{Config.DEBUG_PREFIX} Failed to clear processing message: {clear_error}")
                    pass  # Ignore errors if event_emitter fails

            # Also clear processing_shown flag if it exists to prevent any future issues
            self.processing_shown = False

            end_time = time.time()
            elapsed_time = end_time - self.start_time

            # Skip "Computing" message for faster completion

            model = self._get_auto_router_model(body)

            # Safe output token counting
            try:
                output_content = get_last_assistant_message(body["messages"])

                # Limit content size to prevent hanging
                max_content_size = 5 * 1024 * 1024  # 5MB
                if len(output_content) > max_content_size:
                    if Config.DEBUG:
                        print(
                            f"{Config.DEBUG_PREFIX} Output content too large ({len(output_content)} chars), truncating for token count"
                        )
                    output_content = output_content[:max_content_size]

                # Fast token counting without tiktoken (prevents UI hanging)
                word_count = len(output_content.split())
                output_tokens = int(word_count * 1.3)
                if Config.DEBUG:
                    print(f"{Config.DEBUG_PREFIX} Fast output token count: {word_count} words -> {output_tokens} tokens")

            except Exception as output_error:
                if Config.DEBUG:
                    print(f"{Config.DEBUG_PREFIX} Error processing output tokens: {output_error}")
                # Fallback to zero if everything fails
                output_tokens = 0

            # Check if this is a local model and skip cost calculation if enabled
            if self.valves.skip_free_models and self._is_local_model(model, body):
                tokens = self.input_tokens + output_tokens

                # For local models, just show time and tokens without cost
                elapsed_seconds = int(round(elapsed_time))
                stats = f"{elapsed_seconds} seconds and {tokens} tokens used"

                await __event_emitter__(
                    {"type": "status", "data": {"description": stats, "done": True}}
                )
                return body

            # Skip "Computing" message for faster completion

            total_cost = self.cost_calculator.calculate_costs(
                model, self.input_tokens, output_tokens, self.valves.compensation
            )

            if __user__:
                if "email" in __user__:
                    user_email = __user__["email"]
                else:
                    print("**ERROR: User email not found!")
                try:
                    self.user_cost_manager.update_user_cost(
                        user_email,
                        model,
                        self.input_tokens,
                        output_tokens,
                        total_cost,
                    )
                except Exception as cost_update_error:
                    print("**ERROR: Unable to update user cost file!")
            else:
                print("**ERROR: User not found!")

            tokens = self.input_tokens + output_tokens
            tokens_per_sec = tokens / elapsed_time

            # Convert USD to EUR (approximate rate: 1 USD = 0.93 EUR)
            total_cost_eur = float(total_cost) * 0.93

            # Check if cost should be hidden (for zero or very small costs)
            should_hide_cost = self.valves.hide_zero_cost and (total_cost_eur == 0 or total_cost_eur < 0.00001)

            if self.valves.use_natural_format:
                # Natural language format: "8 seconds, 3395 tokens and 0 € used"
                elapsed_seconds = int(round(elapsed_time))
                if should_hide_cost:
                    stats = f"{elapsed_seconds} seconds and {tokens} tokens used"
                else:
                    cost_str = self._format_euros(total_cost_eur)
                    stats = (
                        f"{elapsed_seconds} seconds, {tokens} tokens and {cost_str} used"
                    )

            await __event_emitter__(
                {"type": "status", "data": {"description": stats, "done": True}}
            )

            return body

        except Exception as outlet_error:
            # Comprehensive error handling to ensure we always show something
            if Config.DEBUG:
                print(f"{Config.DEBUG_PREFIX} Outlet error: {outlet_error}")

            # Calculate basic stats as fallback
            try:
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                elapsed_seconds = int(round(elapsed_time))
                tokens = getattr(self, 'input_tokens', 0)

                # Ensure processing message is cleared
                self.processing_shown = False

                # Show basic stats even if everything else failed
                stats = f"{elapsed_seconds} seconds and {tokens} tokens used (error in cost calculation)"

                await __event_emitter__(
                    {"type": "status", "data": {"description": stats, "done": True}}
                )
            except Exception as fallback_error:
                if Config.DEBUG:
                    print(f"{Config.DEBUG_PREFIX} Fallback error: {fallback_error}")
                # Last resort - just clear the processing message
                try:
                    self.processing_shown = False
                    await __event_emitter__(
                        {"type": "status", "data": {"description": "Cost calculation completed", "done": True}}
                    )
                except Exception:
                    pass  # Give up gracefully

            return body
