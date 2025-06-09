### 0.4.0: 2025-01-27

* Add `hide_zero_cost` valve setting (enabled by default) to hide cost display when cost is zero
* When zero cost is hidden, display format becomes "X seconds and Y tokens used" instead of "X seconds, Y tokens and 0 € used"
* Setting is configurable via valve for users who want to always show cost

### 0.3.9: 2025-01-27

* Add timeout mechanism to auto-hide "Processing X input tokens..." message after 30 seconds
* Prevent status messages from getting stuck on external/piped models
* Import asyncio for timeout functionality

### 0.3.8: 2025-06-09

* Update to the latest version
* Open CHANGELOG.md