import sys
from pathlib import Path

# Ensure app root is in path early, so that submodules (and plugins) can
# resolve ``from core.…`` / ``from services.…`` / ``from routers.…`` imports.
_app_dir = str(Path(__file__).parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

