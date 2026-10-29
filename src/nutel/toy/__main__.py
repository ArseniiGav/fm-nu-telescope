"""``python -m nutel.toy``: the same as the ``nutelsim`` command."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
