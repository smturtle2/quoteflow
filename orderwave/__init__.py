"""Public package interface for `orderwave`.

The library intentionally keeps the external surface minimal:

```python
from orderwave import Market
```
"""

from orderwave.market import Market

__all__ = ["Market"]
