"""CLI: python -m fpo_training_receipts doctor <log_dir>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fpo_training_receipts.receipt import build_receipt, write_receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FPO++ Training Receipts doctor")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Analyze a training run for cliff patterns")
    doctor.add_argument("log_dir", type=Path, help="Path to run log directory")
    doctor.add_argument(
        "--write",
        action="store_true",
        help="Write training_receipt.json to log directory",
    )

    args = parser.parse_args(argv)

    if args.command == "doctor":
        log_dir = args.log_dir.expanduser().resolve()
        if not log_dir.is_dir():
            print(f"error: not a directory: {log_dir}", file=sys.stderr)
            return 1

        receipt = build_receipt(log_dir)
        print(json.dumps(receipt, indent=2))

        if args.write:
            path = write_receipt(log_dir)
            print(f"\nWrote {path}", file=sys.stderr)

        status = receipt["cliff"]["status"]
        return 0 if status in ("healthy", "insufficient_data") else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
