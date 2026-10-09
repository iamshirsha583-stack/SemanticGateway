import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "semantic_gateway") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "semantic_gateway"))

from semantic_gateway.app.main import app
