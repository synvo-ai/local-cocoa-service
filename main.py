#!/usr/bin/env python3
"""
Entry point for PyInstaller-packaged Local Cocoa backend.
This allows the FastAPI app to run as a standalone executable.
"""
import os

def main():
    import uvicorn
    import sys
    from pathlib import Path

    # Get the directory where main.py is located
    project_root = Path(__file__).resolve().parent

    # Ensure project root is in path for 'plugins.xxx' imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    isDebug = os.getenv("DEBUG", "false") == "true"
    host = os.getenv("LOCAL_RAG_HOST", "127.0.0.1")
    port = int(os.getenv("LOCAL_RAG_PORT", "8890"))

    if isDebug and not getattr(sys, 'frozen', False):  # Only reload when not frozen and in debug mode
        uvicorn.run("app.app:app", host=host, port=port, reload=True)
    else:
        from app.app import app as fastapi_app
        uvicorn.run(fastapi_app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
