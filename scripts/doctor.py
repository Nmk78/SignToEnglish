"""Environment report. Same output as `s2e doctor`; run with `uv run python scripts/doctor.py`."""

import sys

from signtoenglish.cli import main

if __name__ == "__main__":
    sys.exit(main(["doctor", *sys.argv[1:]]))
