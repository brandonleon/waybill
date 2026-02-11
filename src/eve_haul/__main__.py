from __future__ import annotations

import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from eve_haul.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    main()
