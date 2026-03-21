from __future__ import annotations

import os
import pathlib
import sys


ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from workspace.eval_libero90_rtc_online import main


if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    main()
