from __future__ import annotations

import argparse
import importlib
import sys

from . import __version__, list_submodules


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cdi-dislo",
        description="CDI_DISLO debugging utilities",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version",
    )

    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="List available submodules",
    )

    parser.add_argument(
        "--import-modules",
        action="store_true",
        help="Import submodules (debug import check)",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Walk subpackages recursively",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue importing even if a module fails (report failures at end)",
    )

    args = parser.parse_args()

    # ------------------------
    # Version
    # ------------------------
    if args.version:
        print(__version__)
        return

    # ------------------------
    # List modules
    # ------------------------
    if args.list_modules:
        modules = list_submodules(recursive=args.recursive)
        for m in modules:
            print(m)
        return

    # ------------------------
    # Import modules (debug)
    # ------------------------
    if args.import_modules:
        modules = list_submodules(recursive=args.recursive)

        successful = []
        failed = []

        for module in modules:
            try:
                importlib.import_module(module)
                successful.append(module)
            except Exception as e:
                failed.append((module, repr(e)))

                if not args.continue_on_error:
                    print(f"FAILED importing {module}")
                    print(f"Error: {e}")
                    sys.exit(1)

        print(f"Imported {len(successful)} modules successfully.")

        if failed:
            print(f"\nFailed imports: {len(failed)}")
            for module, err in failed:
                print(f"- {module}: {err}")
            sys.exit(1)

        return

    parser.print_help()
    sys.exit(0)
