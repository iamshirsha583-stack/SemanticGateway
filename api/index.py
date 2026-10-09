import os
import sys
from pathlib import Path

# Add project root and semantic_gateway to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "semantic_gateway") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "semantic_gateway"))

# Import FastAPI application instance directly
from semantic_gateway.app.main import app

# Explicit top-level FastAPI instance for Vercel Python runtime AST detection
app = app

try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="auto")
except ImportError:
    handler = app

__all__ = ["app", "handler"]
