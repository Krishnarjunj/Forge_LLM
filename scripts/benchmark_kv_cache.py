"""Benchmark generation tokens/sec with KV cache off vs on (M11).

Thin wrapper around ``forge_llm.bench.main`` -- the shared implementation
backs both ``python scripts/benchmark_kv_cache.py`` and the ``forge-llm
bench-cache`` console subcommand.
"""

from __future__ import annotations

import sys

from forge_llm.bench import main

if __name__ == "__main__":
    sys.exit(main())
