#!/usr/bin/env python3
"""
Entry point for PyInstaller-packaged Local Cocoa backend.
This allows the FastAPI app to run as a standalone executable.
"""
import os

def main():
    import uvicorn
    import sys
    import os
    from pathlib import Path

    # Remote debug support: Start as early as possible. so not to use settings here
    debug_port = os.getenv("LOCAL_SERVICE_DEBUG_PORT",'')
    if debug_port and getattr(sys, 'frozen', False):
        try:
            import debugpy
            # If we are in a frozen app, debugpy needs a real python interpreter to run its adapter
            debugpy_python = os.getenv("DEBUGPY_PYTHON_PATH")
            if debugpy_python:
                print(f"[debug] Configuring debugpy to use python: {debugpy_python}")
                debugpy.configure(python=debugpy_python)
            
            print(f"[debug] Debugpy starts listening on 0.0.0.0:{int(debug_port)}")
            debugpy.listen(("0.0.0.0", int(debug_port)))
            if os.getenv("LOCAL_SERVICE_DEBUG_WAIT") == "true":
                print("[debug] Waiting for debugger to attach...")
                debugpy.wait_for_client()
        except Exception as e:
            print(f"[debug] Failed to start debugger: {e}")

    # Determine project root based on whether we are frozen or not
    if getattr(sys, 'frozen', False):
        # In PyInstaller bundle, sys._MEIPASS is the bundle dir
        # For onedir, this is also the directory where the exe and external folders live
        project_root = Path(sys._MEIPASS).resolve()
        
        # Ensure the directory containing the executable is in path (for other binary lookups)
        exe_dir = Path(sys.executable).parent
        if str(exe_dir) not in sys.path:
            sys.path.insert(0, str(exe_dir))
        
        # DEV HACK: If we are in dev mode and the src directory is available, use the src directory
        # This makes debugpy report the original source paths, fixing debugger issues.
        potential_src_root = exe_dir.parent.parent
        if os.getenv("DEBUG") == "true" and (potential_src_root / "app").is_dir():
             print(f"[debug] Dev mode detected, using original source root: {potential_src_root}")
             project_root = potential_src_root

        print(f"[info] Executable dir: {exe_dir}")
    else:
        project_root = Path(__file__).resolve().parent

    # Ensure project root is in path for 'app' and 'plugins' imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"[info] Project root: {project_root}, Frozen: {getattr(sys, 'frozen', False)}")

    from app.core.config import settings

    if settings.is_dev and not getattr(sys, 'frozen', False):  # Only reload when not frozen and in debug mode
        uvicorn.run("app.app:app", host=settings.main_host, port=settings.main_port, reload=True)
    else:
        uvicorn.run("app.app:app", host=settings.main_host, port=settings.main_port, reload=False)


if __name__ == "__main__":
    main()
