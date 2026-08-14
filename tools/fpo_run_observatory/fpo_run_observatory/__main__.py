"""CLI: python -m fpo_run_observatory scan <logs_root>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fpo_run_observatory.report import write_json_report, write_report
from fpo_run_observatory.scan import scan_logs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FPO++ Run Observatory")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan training logs and generate reports")
    scan.add_argument("logs_root", type=Path, help="Log root or single run directory")
    scan.add_argument(
        "--output-dir",
        type=Path,
        default=Path("observatory_out"),
        help="Directory for index.html and report.json",
    )

    args = parser.parse_args(argv)

    if args.command == "scan":
        runs = scan_logs(args.logs_root)
        if not runs:
            print(f"No FPO runs found under {args.logs_root}", file=sys.stderr)
            return 1

        html_path = write_report(runs, args.output_dir / "index.html")
        json_path = write_json_report(runs, args.output_dir / "report.json")
        print(f"Scanned {len(runs)} run(s)")
        print(f"  HTML: {html_path}")
        print(f"  JSON: {json_path}")

        red = sum(1 for r in runs if r.get("grade", {}).get("status") == "red")
        if red:
            return 2
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
