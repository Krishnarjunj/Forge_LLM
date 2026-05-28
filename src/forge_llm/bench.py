"""KV-cache speedup benchmark (M11/M12).

Differentiation move #2: shows the per-step cost of re-computing K and V from
scratch (``use_cache=False``) versus the gpt-fast indexed-cache path
(``use_cache=True``). The CLI is shared by ``scripts/benchmark_kv_cache.py``
and the ``forge-llm bench-cache`` console subcommand (which is the install-
once API).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from forge_llm.config import PRESETS, ForgeConfig
from forge_llm.generation import generate
from forge_llm.model import ForgeForCausalLM
from forge_llm.tokenizer import BPETokenizer

logger = logging.getLogger("forge_llm.bench")

_DEFAULT_CTX_LENGTHS: list[int] = [128, 512, 2048]
_DEFAULT_OUT: Path = Path("docs/results/kv_cache_bench.md")


def _bench(
    model: ForgeForCausalLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new: int,
    use_cache: bool,
    device: str,
) -> float:
    """Generate ``max_new`` tokens; return tokens/sec."""
    start = time.perf_counter()
    tokens = 0
    for _ in generate(
        model,
        tokenizer,
        prompt,
        max_new=max_new,
        seed=0,
        use_cache=use_cache,
        device=device,
    ):
        tokens += 1
    elapsed = time.perf_counter() - start
    return tokens / elapsed if elapsed > 0 else float("inf")


def main(argv: list[str] | None = None) -> int:
    """``forge-llm bench-cache`` and ``python scripts/benchmark_kv_cache.py`` entrypoint."""
    parser = argparse.ArgumentParser(
        description="Benchmark KV-cache speedup at several context lengths."
    )
    parser.add_argument(
        "--ctx-lengths",
        type=int,
        nargs="+",
        default=_DEFAULT_CTX_LENGTHS,
        help=f"Target context lengths (default: {_DEFAULT_CTX_LENGTHS}).",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=None,
        help="Override the model's max_seq (default: max of --ctx-lengths).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup generations before timing (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Markdown output path (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a tiny config (cheap CPU smoke); default uses forge-30m.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    max_seq = args.max_seq if args.max_seq is not None else max(args.ctx_lengths)
    if args.small:
        config = ForgeConfig(
            name="bench-tiny",
            n_layer=2,
            d_model=32,
            n_head=4,
            n_kv_head=2,
            head_dim=8,
            d_ff=64,
            vocab_size=512,
            max_seq=max_seq,
            rope_theta=10000.0,
            norm_eps=1e-5,
            tie_embeddings=True,
            init_std=0.02,
            dropout=0.0,
        )
    else:
        preset = dict(PRESETS["forge-30m"])
        preset["max_seq"] = max_seq
        config = ForgeConfig(**preset)

    logger.info(
        "config: %s (n_layer=%d, d_model=%d, max_seq=%d)",
        config.name,
        config.n_layer,
        config.d_model,
        config.max_seq,
    )

    model = ForgeForCausalLM(config).to(args.device).eval()
    tokenizer = BPETokenizer.train(
        "the quick brown fox jumps over the lazy dog. " * 80,
        vocab_size=config.vocab_size,
    )

    rows: list[tuple[int, float, float, float]] = []
    for ctx in args.ctx_lengths:
        if ctx > config.max_seq:
            logger.warning("ctx=%d exceeds max_seq=%d; skipping", ctx, config.max_seq)
            continue
        max_new = min(ctx, config.max_seq - 4)
        logger.info("ctx=%d (max_new=%d)", ctx, max_new)
        for _ in range(args.warmup):
            _bench(model, tokenizer, "the", max_new=8, use_cache=True, device=args.device)
        nocache = _bench(
            model, tokenizer, "the", max_new=max_new, use_cache=False, device=args.device
        )
        cached = _bench(
            model, tokenizer, "the", max_new=max_new, use_cache=True, device=args.device
        )
        speedup = cached / nocache if nocache > 0 else float("inf")
        rows.append((ctx, nocache, cached, speedup))
        logger.info(
            "  no-cache: %.1f tok/s  cached: %.1f tok/s  speedup: %.2fx",
            nocache,
            cached,
            speedup,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# KV-cache speedup benchmark\n")
    lines.append(
        f"Config: `{config.name}` "
        f"(n_layer={config.n_layer}, d_model={config.d_model}, "
        f"max_seq={config.max_seq}), device=`{args.device}`.\n"
    )
    lines.append(
        "Differentiation move #2: shows the cost of re-computing K/V on every "
        "step (no-cache) versus the gpt-fast pattern (static cache, indexed "
        "writes). Numbers are local; production T4 numbers land in Phase F.\n"
    )
    lines.append("| ctx | no-cache tok/s | cached tok/s | speedup |")
    lines.append("|----:|---------------:|-------------:|--------:|")
    for ctx, nocache, cached, speedup in rows:
        lines.append(f"| {ctx} | {nocache:.1f} | {cached:.1f} | {speedup:.2f}x |")
    lines.append("")
    args.out.write_text("\n".join(lines))
    logger.info("wrote %s", args.out)
    return 0
