import os
import sys
from pathlib import Path

# Resolve base directories and ensure Python modules can be found in Vercel serverless environment
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "semantic_gateway"))
sys.path.insert(0, str(ROOT_DIR / "Semantic Gateway" / "semantic_gateway"))

# Import FastAPI application instance
try:
    from semantic_gateway.app.main import app
except ImportError:
    from app.main import app

# Export ASGI app instance for Vercel Python runtime
# Mangum handler is also provided for AWS Lambda / serverless adapters
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="auto")
except ImportError:
    handler = app

# Ensure 'app' is exposed at module level
__all__ = ["app", "handler"]
