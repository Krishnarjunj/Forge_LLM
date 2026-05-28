"""WikiText-103 (or local-corpus) perplexity evaluation entrypoint (M10).

Thin wrapper around ``forge_llm.eval.main`` so the ``forge-llm eval``
console subcommand and ``python scripts/eval_perplexity.py`` share the same
CLI logic (the implementation lives in the installed package, not in the
script tree).
"""

from __future__ import annotations

import sys

from forge_llm.eval import main

if __name__ == "__main__":
    sys.exit(main())
