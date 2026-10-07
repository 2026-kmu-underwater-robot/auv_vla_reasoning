"""Real-vehicle KMU26 inference runtime (Python 3.10 or newer)."""

import sys

if sys.version_info < (3, 10):
    raise RuntimeError("auv_vla_reasoning requires Python 3.10 or newer")
