### 1.2.6: 2025-06-11

* Fix persistent "Processing X input tokens..." message getting stuck with non-streaming APIs (OpenRouter, piped models)
* Add robust timeout mechanism (10 seconds) as backup to auto-clear stuck processing messages
* Force-clear processing_shown flag in outlet method to prevent future issues
* Enhanced debug logging to track message clearing behavior
* Comprehensive fix addressing root cause: stream() method never called for non-streaming APIs

### 1.2.5: 2025-06-10

* Remove timeout-based approach and ensure processing message is always cleared in outlet function regardless of response time

### 1.2.4: 2025-06-09

* Increase processing message timeout from 5 to 15 seconds for slower non-streaming models
* Add explicit cleanup in outlet function to clear stuck processing messages
* Fix issue where "Processing X input tokens..." message gets stuck with pipe-based or non-streaming models

### 1.2.3: 2025-06-09

* Fix token counting to include ALL messages (system prompts, user messages, assistant messages)
* Replace `get_messages_content()` with manual extraction to ensure complete content capture
* Properly count tokens from entire conversation context including system prompts

### 1.2.2: 2025-06-09

* Add comprehensive debug logging for token counting to help diagnose issues with long prompts
* Debug output shows raw content length, processed content length, and token encoding details
* Help identify if content is being truncated or miscounte

### 1.2.1: 2025-06-09

* Remove non-existent `__model__` parameter from filter function signatures per Open WebUI documentation
* Fix local model detection to work with model name patterns instead of non-existent server metadata
* Update detection logic to identify Ollama and local models based on naming conventions

### 1.2.0: 2025-06-09

* Improve local model detection to focus on serving infrastructure instead of model names
* Detect local inference engines (Ollama, LocalAI, LlamaCpp, KoboldCpp, etc.) in base_url
* Check model metadata for Ollama and other local indicators
* More conservative approach - only skip cost tracking for clearly local models
* Remove unreliable pricing-based detection that caused false positives

### 1.1.1: 2025-06-09

* Increase fallback timeout from 1 second to 5 seconds for non-streaming models

### 1.1.0: 2025-06-09

* Add proper stream hook to detect when AI starts streaming responses
* Processing message now disappears immediately when streaming begins (not timeout-based)
* Use stream() function to cancel timeout task when first stream chunk arrives
* Keep fallback timeout for edge cases where streaming doesn't occur

### 1.0.2: 2025-06-09

* Fix processing message timeout to 1 second (was incorrectly trying to detect streaming)
* Remove outlet-based cancellation logic (outlet runs after streaming, not during)
* Processing message now disappears after 1 second since token counting is nearly instant

### 1.0.1: 2025-06-09

* Detect when AI starts responding and immediately hide "Processing X input tokens..." message
* Cancel timeout task when outlet function begins (response has started)
* Much more responsive UX - message disappears as soon as streaming begins

### 1.0.0: 2025-06-09

* 🎉 **Major Release**: Smart local model detection for cost tracking
* Add `skip_free_models` valve setting (enabled by default) to automatically detect and skip local models
* Check model `base_url` for local IP addresses (127.0.0.1, localhost, private ranges)
* Check model metadata for explicit `local` flag
* Check OpenRouter API for missing pricing data (indicates local model)
* Check for zero-cost models in pricing data
* Local models display: "X seconds and Y tokens used"
* Automatic detection works for Ollama, local inference servers, and other self-hosted models
* Configurable via valve - can be disabled to track all models
* Clean up status message format for local models

### 0.4.0: 2025-06-09

* Add `hide_zero_cost` valve setting (enabled by default) to hide cost display when cost is zero
* When zero cost is hidden, display format becomes "X seconds and Y tokens used"
* Previously showed "X seconds, Y tokens and 0 € used" 
* Setting is configurable via valve for users who want to always show cost

### 0.3.9: 2025-06-09

* Add timeout mechanism to auto-hide "Processing X input tokens..." message after 30 seconds
* Prevent status messages from getting stuck on external/piped models
* Import asyncio for timeout functionality

### 0.3.8: 2025-06-09

* Update to the latest version
* Open CHANGELOG.md