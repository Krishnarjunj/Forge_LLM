"""``forge-llm`` console entrypoint (M12).

Four subcommands -- ``generate``, ``train``, ``eval``, ``bench-cache`` -- all
dispatched via stdlib argparse (ADR-003: argparse over tyro for subcommand
ergonomics and zero-dep).

* ``generate`` is implemented here directly: load a Forge checkpoint via
  ``hub.from_pretrained``, stream tokens with ``generate``, print to stdout.
* ``train`` delegates to ``forge_llm.train.main``.
* ``eval`` delegates to ``forge_llm.eval.main``.
* ``bench-cache`` delegates to ``forge_llm.bench.main``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge_llm import bench as _bench
from forge_llm import eval as _eval
from forge_llm import train as _train
from forge_llm.generation import generate
from forge_llm.hub import from_pretrained
from forge_llm.tokenizer import BPETokenizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge-llm",
        description="Forge-LLM: ~30M decoder-only LM trained from scratch on free Kaggle T4.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<cmd>")

    gen = sub.add_parser("generate", help="Generate text from a checkpoint.")
    gen.add_argument("prompt", help="Prompt text.")
    gen.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory written by `save_pretrained` (or `from_pretrained`).",
    )
    gen.add_argument("--max-new", type=int, default=50)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--rep-penalty", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the KV cache (mostly for the speedup benchmark).",
    )
    gen.add_argument("--device", default="cpu")

    sub.add_parser(
        "train",
        help="Train a Forge model. Forwards all flags to forge_llm.train.",
        add_help=False,
    )
    sub.add_parser(
        "eval",
        help="Evaluate byte-perplexity. Forwards all flags to forge_llm.eval.",
        add_help=False,
    )
    sub.add_parser(
        "bench-cache",
        help="KV-cache speedup benchmark. Forwards all flags to forge_llm.bench.",
        add_help=False,
    )
    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    tokenizer = BPETokenizer.load(args.checkpoint / "tokenizer.json")
    # from_pretrained needs a data_factory for the Trainer's stateful dataset,
    # but generate() only touches model.* -- supply an empty stub stream.
    trainer = from_pretrained(args.checkpoint, data_factory=lambda: iter([]), device=args.device)
    for piece in generate(
        trainer.model,
        tokenizer,
        args.prompt,
        max_new=args.max_new,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        rep_penalty=args.rep_penalty,
        seed=args.seed,
        use_cache=not args.no_cache,
        device=args.device,
    ):
        sys.stdout.write(piece)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Subcommands that forward all flags are dispatched by splitting argv at the
    # first positional. ``--help`` on a forwarded subcommand is handled by its
    # own parser inside the target module's main().
    if argv and argv[0] in {"train", "eval", "bench-cache"}:
        forward = argv[1:]
        if argv[0] == "train":
            return _train.main(forward)
        if argv[0] == "eval":
            return _eval.main(forward)
        return _bench.main(forward)

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "generate":
        return _cmd_generate(args)
    parser.error(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
