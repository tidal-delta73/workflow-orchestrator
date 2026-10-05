"""Command line entry point: version, help, and plan."""
import sys

from . import __version__
from . import plan as plan_module

USAGE = """usage: python3 -m workflow_orchestrator <command>

commands:
  version                  print the package version
  help                     print this message
  plan <definition.json>   print static execution levels for a workflow
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "version":
        print(__version__)
        return 0
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    if command == "plan":
        if len(args) != 2:
            print(USAGE, end="", file=sys.stderr)
            return 2
        return plan_module.run(args[1], sys.stdout, sys.stderr)
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
