"""Sync runtime/ from sibling rsna-knee project."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RSNA_KNEE = HERE.parent / "rsna-knee"
SYNC_TOOL = RSNA_KNEE / "tools" / "sync_runtime.py"


def main() -> None:
    if not RSNA_KNEE.is_dir():
        print(f"rsna-knee not found at {RSNA_KNEE}", file=sys.stderr)
        sys.exit(1)
    if not SYNC_TOOL.is_file():
        print(f"sync tool missing: {SYNC_TOOL}", file=sys.stderr)
        sys.exit(1)

    subprocess.run([sys.executable, str(SYNC_TOOL)], check=True, cwd=RSNA_KNEE)
    print(f"done -> {HERE / 'runtime'}")


if __name__ == "__main__":
    main()
