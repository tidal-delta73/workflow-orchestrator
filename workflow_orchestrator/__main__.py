"""Command line entry point: `version`, `help` and `plan`."""
import json
import sys

from . import __version__
from .planner import DefinitionError, plan_levels

USAGE = """usage: python3 -m workflow_orchestrator <command>

commands:
  version                 print the package version
  plan <definition.json>  validate a workflow definition and print its
                          execution levels as a single JSON line
  help                    print this message
"""


def _run_plan(args: list[str]) -> int:
    if len(args) != 1:
        print(USAGE, end="", file=sys.stderr)
        return 2
    path = args[0]
    try:
        with open(path, "rb") as stream:
            raw = stream.read()
    except OSError:
        print(f"cannot read definition: {path}", file=sys.stderr)
        return 1
    try:
        definition = json.loads(raw)
    except json.JSONDecodeError:
        print("invalid json", file=sys.stderr)
        return 2
    try:
        levels = plan_levels(definition)
    except DefinitionError as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.write(
        json.dumps({"levels": levels}, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "version":
        print(__version__)
        return 0
    if command == "plan":
        return _run_plan(args[1:])
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
