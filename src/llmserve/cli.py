"""Command-line entry point: ``llmserve <command> ...``."""

from __future__ import annotations

import sys

COMMANDS = {
    "fetch": ("llmserve.models.hub", "download models by alias"),
    "bench": ("llmserve.bench.run", "run a benchmark"),
    "generate": ("llmserve.generate_cli", "generate text from the command line"),
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: llmserve <command> [args]\n\ncommands:")
        for name, (_, helptext) in COMMANDS.items():
            print(f"  {name:10s} {helptext}")
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 1)
    import importlib

    mod = importlib.import_module(COMMANDS[argv[0]][0])
    mod.main(argv[1:])


if __name__ == "__main__":
    main()
