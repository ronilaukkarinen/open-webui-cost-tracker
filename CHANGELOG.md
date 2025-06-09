### 1.0.0: 2025-01-27

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
* Remove "(local model)" text from status display

### 0.4.0: 2025-01-27

* Add `hide_zero_cost` valve setting (enabled by default) to hide cost display when cost is zero
* When zero cost is hidden, display format becomes "X seconds and Y tokens used"
* Previously showed "X seconds, Y tokens and 0 € used" 
* Setting is configurable via valve for users who want to always show cost

### 0.3.9: 2025-01-27

* Add timeout mechanism to auto-hide "Processing X input tokens..." message after 30 seconds
* Prevent status messages from getting stuck on external/piped models
* Import asyncio for timeout functionality

### 0.3.8: 2025-06-09

* Update to the latest version
* Open CHANGELOG.md