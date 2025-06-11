### 1.2.14: 2025-06-11

* Add automatic cost tracking support for Auto Router (works by default, no valve needed)
* Prevent Auto Router from being treated as a local/free model - now attempts cost calculation
* Add comprehensive model detection for Auto Router by checking response metadata and message fields
* Implement intelligent fallback pricing using common models (GPT-3.5, Claude Haiku, Gemini Pro) when exact model can't be detected
* Enable cost tracking for Auto Router instead of showing only "X seconds and Y tokens used"
* Auto Router detection works automatically without configuration

### 1.2.13: 2025-06-11

* Completely rewrite outlet method to fix persistent scoping errors and stuck processing messages
* Move format_euros function outside outlet method to eliminate nested function scoping issues
* Add comprehensive error handling with multiple fallback levels to ensure messages are always cleared
* Fix issue where "Processing X input tokens..." gets stuck with shorter token requests
* Add robust exception handling to guarantee cost tracking always shows a result
* Eliminate all nested function variable capture that caused "cannot access free variable 'e'" errors

### 1.2.12: 2025-06-11

* Remove timeout mechanism completely to eliminate persistent scoping errors
* Simplify inlet method by removing problematic nested async function architecture  
* Fix "cannot access free variable 'e'" error by eliminating complex closure variable capture
* Processing messages will now only be cleared by outlet method and stream method (no timeout fallback)
* Cleaner, more reliable code without nested function scoping complications

### 1.2.11: 2025-06-11

* Fix remaining scoping issues: replace `except Exception as _:` with proper variable name
* Add explicit variable capture in timeout function to avoid closure-related scoping errors  
* Capture `self.input_tokens` as local variable `input_tokens` in nested function to prevent variable access issues
* Final resolution of "cannot access free variable 'e'" error through comprehensive scoping fixes

### 1.2.10: 2025-06-11

* Fix persistent Python scoping error by refactoring nested async function architecture
* Move timeout mechanism from nested inline function to separate method to avoid variable capture issues
* Replace problematic nested function scope with explicit parameter passing for event_emitter
* Eliminate Python variable scoping conflicts that caused "cannot access free variable" errors

### 1.2.9: 2025-06-11

* Fix Python scoping error "cannot access free variable 'e' where it is not associated with a value in enclosing scope"
* Rename exception variable in nested async function to avoid scoping conflicts with outer scope variables

### 1.2.8: 2025-06-11

* Fix cost hiding threshold from < 0.0001 € to < 0.00001 € for more precise zero cost detection
* Add automatic streaming disable for large requests (>10k tokens by default) to ensure cost tracking works properly
* Add `disable_streaming_large_requests` valve setting (enabled by default) to control streaming behavior for large requests
* Add `large_request_token_threshold` valve setting (default: 10000) to configure when streaming should be disabled
* Fix issue where cost tracker would show nothing for very large token requests due to streaming issues

### 1.2.7: 2025-06-11

* Fix hide_zero_cost setting not working properly - was showing "X seconds, Y tokens and 0 € used" instead of "X seconds and Y tokens used"
* Improve zero cost detection to include very small costs (< 0.0001 €) that should be hidden
* Restructure cost formatting logic to only format cost string when actually displaying it
* Enhanced hide_zero_cost logic to work correctly with natural language format

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