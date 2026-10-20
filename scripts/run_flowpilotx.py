#!/usr/bin/env python3
"""Start the local FlowPilotX application without installing the package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from khu_sulfoxidation.gui_server import main

if __name__ == "__main__":
    main()
