#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import platform
import time
import threading
import argparse  # Added for argument parsing
from pathlib import Path
from typing import Optional, List, Dict, Literal
import zipfile

# --------------------------
# Configuration & Constants
# --------------------------
DEFAULT_PYTHON_VERSION = "3.12.12"
SPINNER_CHARS = "|/-\\"
PLATFORM_SYSTEM = platform.system()  # Renamed to avoid conflict with arg
IS_WINDOWS = PLATFORM_SYSTEM == "Windows"
IS_LINUX = PLATFORM_SYSTEM == "Linux"
BuildMode = Literal["dev", "prod"]
SupportedPlatforms = Literal["win", "linux", "mac"]  # Added platform type

# --------------------------
# Build Mode Configuration
# --------------------------
def get_build_config(mode: BuildMode) -> Dict:
    """Get environment-specific build configuration"""
    base_config = {
        "collect_modules": [
            "uvicorn", "fastapi", "pydantic", "pydantic_settings", "starlette", "qdrant_client",
            "tiktoken", "rapidocr_onnxruntime", "onnxruntime", "langdetect",
            "certifi", "services", "magika", "markitdown", "grpcio",
            "azure", "msal", "msgraph", "debugpy", "mcp", "sse_starlette", "httpx", "dotenv", "mail"
            "docx", "pptx", "openpyxl", "xlrd", "pypdf", "pdfminer", "mammoth", "lxml",
            "bs4", "pymongo", "sounddevice", "pydub", "speech_recognition",
            "rich", "typer", "fitz", "PIL", "cv2", "numpy"
        ],
        "hidden_imports": [
            "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
            "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
            "email_validator", "multipart", "PIL", "cv2", "numpy",
            "fitz", "xxhash", "markdownify", "markdown_it", "msal",
            "msal_extensions", "azure.identity", "msgraph", "onnx", "email",
            "docx", "pptx", "openpyxl", "xlrd", "pypdf", "pdfminer", "mammoth",
            "imaplib", "poplib", "smtplib", "ssl"
        ],
        "exclude_modules": [
            "tkinter", "matplotlib", "scipy", "pandas", "torch", "tensorflow"
        ],
        "exclude_patterns": [
            "__pycache__", "*.pyc", ".mypy_cache", ".pytest_cache",
            ".DS_Store", "*.spec"
        ],
        "clean_build": True,
        "include_debug_symbols": False,
        "optimize_level": 0
    }

    # Dev mode overrides
    if mode == "dev":
        return {
            **base_config,
            "clean_build": False,  # Keep build artifacts for faster rebuilds
            "include_debug_symbols": True,
            "optimize_level": 0,
            "exclude_modules": base_config["exclude_modules"] + ["pyinstaller", "app", "plugins"],
            "extra_pyinstaller_args": ["--noupx", "--exclude-module", "app", "--exclude-module", "plugins"]  # disable compression under dev mode for faster build performance
        }

    # Prod mode overrides
    elif mode == "prod":
        return {
            **base_config,
            "clean_build": True,
            "include_debug_symbols": False,
            "optimize_level": 2,
            "exclude_modules": base_config["exclude_modules"] + ["debugpy"],
            "extra_pyinstaller_args": [
                "--optimize=2",
                "--strip",  # Remove symbols (Linux/macOS only)
                "--onefile"  # Single executable (prod-only)
            ]
        }

# --------------------------
# Utility Functions
# --------------------------
class Spinner:
    """Simple spinner for long-running operations"""
    def __init__(self, label: str):
        self.label = label
        self.running = False
        self.spinner_thread: Optional[threading.Thread] = None

    def _spin(self):
        """Internal spinner loop"""
        i = 0
        while self.running:
            sys.stdout.write(f"\r[{SPINNER_CHARS[i % 4]}] {self.label}...")
            sys.stdout.flush()
            i += 1
            time.sleep(0.2)

    def start(self):
        """Start the spinner"""
        if not sys.stdout.isatty():  # Disable spinner for non-TTY outputs
            print(f"{self.label}...")
            return
        self.running = True
        self.spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self.spinner_thread.start()

    def stop(self, success: bool):
        """Stop the spinner and show result"""
        if not sys.stdout.isatty():
            return
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        
        status = "✓" if success else "✗"
        sys.stdout.write(f"\r[{status}] {self.label}\n")
        sys.stdout.flush()

def run_with_spinner(label: str, cmd: List[str], cwd: Optional[Path] = None, show_output: bool = False) -> int:
    """
    Run a command with a spinner progress indicator
    
    Args:
        label: Human-readable label for the operation
        cmd: Command to execute (list of strings)
        cwd: Working directory for the command
        show_output: Whether to display command stdout/stderr in real-time
    
    Returns:
        Exit code of the command
    """
    spinner = Spinner(label)
    spinner.start()
    
    try:
        if show_output:
            # Show output in real-time (no capture)
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
        else:
            # Capture output (original behavior)
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        
        spinner.stop(result.returncode == 0)
        if result.returncode != 0 and not show_output:
            print(f"Error running command: {' '.join(cmd)}", file=sys.stderr)
            print(f"STDERR: {result.stderr}", file=sys.stderr)
        return result.returncode
    except Exception as e:
        spinner.stop(False)
        print(f"Command failed with exception: {e}", file=sys.stderr)
        return 1

def find_python_interpreter(requested_version: str = DEFAULT_PYTHON_VERSION, target_platform: SupportedPlatforms = None) -> Path:
    """
    Find or provision a suitable Python interpreter
    
    Args:
        requested_version: Target Python version
        target_platform: Target build platform (for cross-compilation support)
    
    Returns:
        Path to Python executable
    
    Raises:
        RuntimeError: If no suitable interpreter can be found/provisioned
    """
    # Check PYTHON_BIN environment variable first
    python_bin = os.getenv("PYTHON_BIN")
    if python_bin:
        path = Path(python_bin)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise RuntimeError(f"PYTHON_BIN points to non-executable: {python_bin}")

    # Extract major.minor version (e.g. 3.11 from 3.11.11)
    major_minor = requested_version.rsplit(".", 1)[0]
    candidates = []

    # Build platform-specific candidate list (honor target platform if specified)
    current_platform = PLATFORM_SYSTEM
    if target_platform:
        if target_platform == "win":
            current_platform = "Windows"
        elif target_platform == "linux":
            current_platform = "Linux"
        elif target_platform == "mac":
            current_platform = "Darwin"

    if current_platform == "Windows":
        candidates = [
            f"python{major_minor.replace('.', '')}.exe",
            f"python{major_minor}.exe",
            "python3.exe",
            "python.exe"
        ]
    else:
        candidates = [
            f"python{major_minor}",
            f"python{major_minor.replace('.', '')}",
            "python3",
            "python"
        ]

    # Search PATH for candidates
    for candidate in candidates:
        try:
            path = Path(shutil.which(candidate))
            if path:
                # Verify version compatibility (basic check)
                version_output = subprocess.check_output(
                    [str(path), "--version"],
                    text=True,
                    stderr=subprocess.STDOUT
                )
                if major_minor in version_output:
                    return path
        except (subprocess.CalledProcessError, FileNotFoundError, TypeError):
            continue

    # Try pyenv if available (Linux/macOS only)
    if current_platform != "Windows":
        try:
            if shutil.which("pyenv"):
                print(f"Using pyenv to provision Python {requested_version}")
                subprocess.run(
                    ["pyenv", "install", "-s", requested_version],
                    check=True,
                    capture_output=True,
                    text=True
                )
                pyenv_root = subprocess.check_output(
                    ["pyenv", "root"],
                    text=True
                ).strip()
                pyenv_python = Path(pyenv_root) / "versions" / requested_version / "bin" / "python3"
                if pyenv_python.is_file() and os.access(pyenv_python, os.X_OK):
                    return pyenv_python
        except subprocess.CalledProcessError as e:
            print(f"Pyenv installation failed: {e}", file=sys.stderr)

    raise RuntimeError(
        f"No suitable Python {major_minor} interpreter found for {current_platform}.\n"
        "Options:\n"
        f"1. Install Python {major_minor}\n"
        "2. Set PYTHON_BIN environment variable to point to a valid interpreter\n"
        "3. Install pyenv (Linux/macOS) to auto-provision the required version"
    )

def create_venv(python_path: Path, venv_dir: Path, mode: BuildMode, no_cache: bool = False) -> Path:
    """
    Create a virtual environment for building (with cache reuse)
    
    Args:
        python_path: Path to base Python interpreter
        venv_dir: Directory to create venv in
        mode: Build mode (dev/prod)
        no_cache: Whether to force recreate venv (ignore existing)
    
    Returns:
        Path to Python executable in venv
    """
    venv_name = f"venv"
    venv_dir = venv_dir / venv_name
    venv_python_path = venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

    # Check if a ven already exists, re-use if possible no_cache is False
    if not no_cache and venv_dir.exists() and venv_python_path.exists():
        try:
            # version if venv is valid
            result = subprocess.run(
                [str(venv_python_path), "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Reusing existing virtual environment at {venv_dir} (mode: {mode})")
            return venv_python_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Existing venv at {venv_dir} is invalid, recreating...")

    # need to create/ rebuild venv
    if venv_dir.exists():
        print(f"Removing existing virtual environment at {venv_dir} (mode: {mode})")
        shutil.rmtree(venv_dir, ignore_errors=True)

    print(f"Creating {mode} build virtual environment at {venv_dir}")
    venv_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [str(python_path), "-m", "venv", str(venv_dir)]
    # Add dev-specific venv flags
    if mode == "dev":
        cmd.append("--symlinks")  # Faster venv creation (non-Windows)
    
    result = run_with_spinner(f"Creating {mode} virtual environment", cmd)
    if result != 0:
        raise RuntimeError(f"Failed to create {mode} virtual environment")

    # Return path to venv Python executable
    if not venv_python_path.exists():
        raise RuntimeError(f"Virtual environment creation failed: {venv_python_path} not found")
    return venv_python_path

def install_dependencies(venv_python: Path, requirements_path: Path, mode: BuildMode, no_cache: bool = False):
    """
    Install dependencies into virtual environment with mode-specific behavior
    
    Args:
        venv_python: Path to venv Python executable
        requirements_path: Path to requirements.txt
        mode: Build mode (dev/prod)
        no_cache: Whether to force reinstall dependencies (only relevant for existing venv)
    """
    # only install dependencies when force rebuild venv or first create
    # check if need to install dependencies (by checking if pip freeze has content)
    install_deps = True
    if not no_cache:
        try:
            freeze_result = subprocess.run(
                [str(venv_python), "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            # if already has dependencies, skip install (simple check: output is not empty)
            if freeze_result.stdout.strip():
                print(f"Reusing existing dependencies in venv (mode: {mode})")
                install_deps = False
        except subprocess.CalledProcessError:
            install_deps = True

    if not install_deps:
        return

    # Upgrade pip/wheel first
    run_with_spinner(
        f"Upgrading pip and wheel ({mode})",
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "wheel"]
    )

    # Install base requirements
    req_cmd = [str(venv_python), "-m", "pip", "install"]
    if mode == "prod":
        req_cmd.append("--no-cache-dir")  # Reduce size for prod
        req_cmd.append("--upgrade")
    req_cmd.extend(["-r", str(requirements_path)])
    
    run_with_spinner(
        f"Installing project dependencies ({mode})",
        req_cmd,
        show_output=True
    )

    # Install dev-only dependencies if needed
    if mode == "dev":
        dev_requirements = requirements_path.parent / "requirements-dev.txt"
        if dev_requirements.exists():
            run_with_spinner(
                "Installing development dependencies",
                [str(venv_python), "-m", "pip", "install", "-r", str(dev_requirements)],
                show_output=True
            )
        else:
            print("Warning: requirements-dev.txt not found - skipping dev dependencies")

    # Install PyInstaller (version consistent across modes)
    run_with_spinner(
        f"Installing PyInstaller ({mode})",
        [str(venv_python), "-m", "pip", "install", "pyinstaller"]
    )

def copy_source_files(src_dir: Path, dest_dir: Path, exclude_patterns: List[str], mode: BuildMode):
    """
    Copy source files with exclusion patterns and mode-specific behavior
    
    Args:
        src_dir: Source directory
        dest_dir: Destination directory
        exclude_patterns: List of patterns to exclude (glob style)
        mode: Build mode (dev/prod)
    """
    print(f"Copying source files from {src_dir} to {dest_dir} (mode: {mode})")
    shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Add mode-specific exclusions
    mode_exclusions = []
    if mode == "prod":
        mode_exclusions = ["tests", "test_*.py", "*.test.py", "dev-scripts"]
    else:
        mode_exclusions = ["prod-only*"]
    
    all_exclusions = exclude_patterns + mode_exclusions

    for item in src_dir.glob("**/*"):
        # Skip excluded patterns
        exclude = False
        rel_path = item.relative_to(src_dir)
        for pattern in all_exclusions:
            if rel_path.match(pattern):
                exclude = True
                break
            
            for part in rel_path.parts:
                if part == pattern:
                    exclude = True
                    break
        if exclude:
            continue

        # Create destination path
        dest_path = dest_dir / rel_path
        if item.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
        else:
            shutil.copy2(item, dest_path)

def fix_llama_cpp_rpaths(llama_server_path: Path):
    """
    Fix rpath for llama-server (mac/Linux only)
    
    Args:
        llama_server_path: Path to llama-server binary
    """
    if not (IS_LINUX or platform.system() == "Darwin"):
        return

    if not llama_server_path.exists():
        print(f"Warning: llama-server not found at {llama_server_path}", file=sys.stderr)
        return

    try:
        # macOS-specific rpath fixes
        if platform.system() == "Darwin":
            # Remove old build path rpath
            subprocess.run(
                ["install_name_tool", "-delete_rpath", str(llama_server_path.parent.parent / "build" / "bin"), str(llama_server_path)],
                capture_output=True,
                text=True
            )
            # Add relative rpath if missing
            otool_output = subprocess.check_output(["otool", "-l", str(llama_server_path)], text=True)
            if "@executable_path" not in otool_output:
                subprocess.run(
                    ["install_name_tool", "-add_rpath", "@executable_path", str(llama_server_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
        print("Fixed rpath in llama-server")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Failed to fix rpath: {e}", file=sys.stderr)

def build_pyinstaller_bundle(
    venv_python: Path,
    main_path: Path,
    dist_dir: Path,
    build_dir: Path,
    mode: BuildMode,
    target_platform: SupportedPlatforms
):
    """
    Run PyInstaller to create standalone bundle with mode-specific configuration
    
    Args:
        venv_python: Path to venv Python executable
        main_path: Path to main.py entry point
        dist_dir: Output directory for distribution
        build_dir: Working directory for PyInstaller
        mode: Build mode (dev/prod)
        target_platform: Target build platform (win/linux/mac)
    """
    config = get_build_config(mode)
    print(f"Running PyInstaller for {mode} mode (target: {target_platform}) (this may take several minutes)...")
    
    # Build PyInstaller command
    cmd = [
        str(venv_python),
        "-m", "PyInstaller",
        "--name", f"local-cocoa-server",
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir / f"work"),
        "--specpath", str(build_dir),
        "--noconfirm",
        "--log-level", "WARN"
    ]

    # Add mode-specific clean flag
    if config["clean_build"]:
        cmd.append("--clean")

    # Add collect-all modules
    for module in config["collect_modules"]:
        cmd.extend(["--collect-all", module])

    # Add hidden imports
    for imp in config["hidden_imports"]:
        cmd.extend(["--hidden-import", imp])

    # Add exclude modules
    for mod in config["exclude_modules"]:
        cmd.extend(["--exclude-module", mod])

    # Add paths
    cmd.extend(["--paths", str(build_dir)])
    cmd.extend(["--paths", str(build_dir / "app")])
    cmd.extend(["--paths", str(build_dir / "plugins")])

    # Add mode-specific arguments
    cmd.extend(config["extra_pyinstaller_args"])

    # Add data files (env files and config)
    for data_file in [".env", f".env.{mode}", "models.config.json"]:
        if (build_dir / data_file).exists():
            cmd.extend(["--add-data", f"{data_file}{os.pathsep}."])


    # Add platform-specific arguments
    if target_platform == "win":
        # Windows-specific PyInstaller args
        cmd.extend(["--console"])  # Show console (adjust as needed)
    elif target_platform == "linux":
        # Linux-specific PyInstaller args
        cmd.extend(["--strip"])
    elif target_platform == "mac":
        # macOS-specific PyInstaller args
        cmd.extend(["--strip", "--osx-bundle-identifier", "com.localrag.server"])

    # Add entry point
    cmd.append(str(main_path))

    # Run PyInstaller
    result = run_with_spinner(f"Building {mode} standalone bundle (target: {target_platform})", cmd, cwd=build_dir)
    if result != 0:
        raise RuntimeError(f"PyInstaller {mode} build failed for {target_platform}")

def write_readme(dist_dir: Path, python_version: str, mode: BuildMode, target_platform: SupportedPlatforms):
    """
    Write README.md for distribution with mode-specific information
    
    Args:
        dist_dir: Distribution directory
        python_version: Python version used for build
        mode: Build mode (dev/prod)
        target_platform: Target build platform (win/linux/mac)
    """
    readme_path = dist_dir / f"README_{mode}.md"
    print(f"Writing {mode} README to {readme_path}")
    
    mode_notes = {
        "dev": """
## Development Mode Notes
- Includes debug symbols and import tracing
- Preserves build artifacts for faster rebuilds
- Includes development dependencies
- Not optimized for size/performance
- May include additional logging/debug features
""",
        "prod": """
## Production Mode Notes
- Optimized for size and performance (--optimize=2)
- Single-file executable (--onefile)
- Stripped of debug symbols (Linux/mac)
- No development dependencies included
- Clean build (no cached artifacts)
"""
    }

    platform_notes = {
        "win": "## Windows Notes\n- Executable is a .exe file\n- Requires no additional dependencies (PyInstaller onefile)",
        "linux": "## Linux Notes\n- Executable is stripped of debug symbols\n- May require libstdc++ and other system libraries",
        "mac": "## Mac Notes\n- Executable is signed with ad-hoc signature\n- May require Gatekeeper exceptions for unsigned binaries"
    }
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# Local RAG Agent Distribution ({mode.upper()} Mode - {target_platform.upper()})

This directory is generated by the packaging script using Python {python_version}.
It contains a {mode} mode PyInstaller bundle built for {target_platform}.

## How to Run
- {'Windows: `run_{mode}.bat`' if target_platform == 'win' else f'{target_platform.capitalize()}: `./run_{mode}.sh`'}

## Environment Variables
- LOCAL_SERVICE_BIN_PATH: Root path of all the binary exeution files (local-cocoa-server、llama、whisper)
- LOCAL_SERVICE_LOG_TO_FILE: Whether to log to file (default: True)
- LOCAL_SERVICE_MAIN_HOST: Host to bind to (default: 127.0.0.1)
- LOCAL_SERVICE_MAIN_PORT: Port to listen on (default: 8890)
- LOCAL_RUNTIME_ROOT: Data directory for RAG storage
- LOCAL_SERVICE_DEBUG_PORT: Port to listen on for debug (default: 8891). Only effective in dev mode.
- LOCAL_SERVICE_DEBUG_WAIT: Whether to wait for debugger to attach (default: False). Only effective in dev mode.

## Binary Contents
The `llama-cpp/bin` folder contains the llama.cpp server binaries (if available).

{mode_notes[mode]}

{platform_notes[target_platform].format(mode=mode)}
""")

def zip_directory(source_dir: Path, zip_path: Path):
    """
    Compress the source directory to a zip file
    
    Args:
        source_dir: the dir to be compressed
        zip_path: the output zip file path 
    """
    spinner = Spinner(f"Compressing {source_dir} to {zip_path}")
    spinner.start()
    
    try:
        # Ensure the parent directory of the ZIP file exists
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the ZIP file and write its contents
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            # Iterate over all files in the directory
            for file_path in source_dir.rglob('*'):
                # Skip the target zip file itself if it's inside the source directory
                if file_path.resolve() == zip_path.resolve():
                    continue
                    
                # Calculate the relative path (as the path inside the ZIP)
                rel_path = file_path.relative_to(source_dir)
                # Add the file to the ZIP
                zipf.write(file_path, rel_path)
        
        spinner.stop(True)
        print(f"Successfully created ZIP archive: {zip_path}")
        print(f"ZIP file size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        spinner.stop(False)
        raise RuntimeError(f"Failed to create ZIP archive: {e}")

# --------------------------
# Main Build Function
# --------------------------
def main(
    root_dir: Path, 
    mode: BuildMode, 
    target_platform: SupportedPlatforms, 
    output_dir: Path, 
    no_cache: bool = False, 
    no_packing: bool = False,
    incremental: bool = False
):
    """
    Main packaging function with mode and platform support
    
    Args:
        root_dir: Root directory of the project
        mode: Build mode (dev/prod)
        target_platform: Target build platform (win/linux/mac)
        output_dir: Custom output directory for distribution
        no_cache: Whether to force recreate venv and reinstall dependencies
        no_packing: Whether to skip ZIP packing
        incremental: Whether to perform a fast incremental update (source only)
    
    Raises:
        RuntimeError: If any step fails
    """
    config = get_build_config(mode)

    # Directory setup with custom output dir
    dist_dir = output_dir.resolve()
    # In onedir mode, PyInstaller puts everything in a subdirectory
    app_dist_dir = dist_dir / "local-cocoa-server"
    if mode == "prod":
        # In onefile mode, it's just the executable in dist_dir, but we still handle it
        app_dist_dir = dist_dir

    build_dir = root_dir / "build" / target_platform
    requirements_path = root_dir / "app" / "requirements.txt"
    main_py_path = root_dir / "main.py"
    llama_cpp_src = root_dir / "dist" / "llama-cpp"
    whisper_cpp_src = root_dir / "dist" / "whisper-cpp"

    # Fast Path: Incremental Update
    if incremental and mode == "dev":
        exe_ext = ".exe" if target_platform == "win" else ""
        exe_path = app_dist_dir / f"local-cocoa-server{exe_ext}"
        
        if exe_path.exists():
            print(f">>> Performing fast incremental update for {mode} mode...")
            
            # Step 1: Copy source files (mode-specific exclusions)
            # Directly to the distribution directory to overwrite existing files
            print(f"Updating source code in {app_dist_dir}...")
            copy_source_files(root_dir / "app", app_dist_dir / "app", config["exclude_patterns"], mode)
            copy_source_files(root_dir / "plugins", app_dist_dir / "plugins", config["exclude_patterns"], mode)
            
            # Update main.py (if it exists as a file next to exe, though usually it's bundled)
            # If it's bundled, updating the external one won't help unless main.py is excluded too.
            # But the exclusion logic in get_build_config handles 'app' and 'plugins'.
            shutil.copy2(main_py_path, build_dir / "main.py")
            
            # Step 2: Copy .env and config files
            for data_file in [".env", f".env.{mode}", "models.config.json"]:
                src_file = root_dir / data_file
                if src_file.is_file():
                    shutil.copy2(src_file, app_dist_dir / data_file)
            
            print("✓ Incremental update complete.")
            
            # Optional packing
            if not no_packing:
                zip_filename = f"local-cocoa-server-{target_platform}-{mode}.zip"
                zip_path = output_dir.resolve() / zip_filename
                zip_directory(dist_dir, zip_path)
            return

    # Regular Build Path...

    # Step 1: Find Python interpreter (with platform awareness)
    python_path = find_python_interpreter(target_platform=target_platform)
    python_version = subprocess.check_output(
        [str(python_path), "-c", "import platform; print(platform.python_version())"],
        text=True
    ).strip()
    print(f"Using Python {python_version} at {python_path} (mode: {mode}, target: {target_platform})")

    # Step 2: Clean up previous builds (mode-specific)
    if config["clean_build"]:
        print(f"Cleaning up previous {mode} build artifacts for {target_platform}")
        # When cleaning dist_dir, we preserve llama-cpp and whisper-cpp if they exist
        if dist_dir.exists():
            for item in dist_dir.iterdir():
                # Avoid deleting manually placed binary directories
                if item.is_dir() and item.name in ["llama-cpp", "whisper-cpp"]:
                    print(f"Preserving {item.name} in distribution directory")
                    continue
                
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
        
        shutil.rmtree(build_dir / "work", ignore_errors=True)
    
    dist_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Create build venv (with cache reuse)
    venv_python = create_venv(python_path, build_dir, mode, no_cache)

    # Step 4: Install dependencies (with cache reuse)
    if not requirements_path.is_file():
        raise RuntimeError(f"Requirements file not found: {requirements_path}")
    install_dependencies(venv_python, requirements_path, mode, no_cache)

    # Step 5: Copy source files (mode-specific exclusions)
    copy_source_files(root_dir / "app", build_dir / "app", config["exclude_patterns"], mode)
    copy_source_files(root_dir / "plugins", build_dir / "plugins", config["exclude_patterns"], mode)

    # Step 5.1: Copy .env and config files to build directory
    for data_file in [".env", f".env.{mode}", "models.config.json"]:
        src_file = root_dir / data_file
        if src_file.is_file():
            print(f"Copying {data_file} to build directory")
            shutil.copy2(src_file, build_dir / data_file)


    # Copy main.py
    if not main_py_path.is_file():
        raise RuntimeError(f"Main entry point not found: {main_py_path}")
    shutil.copy2(main_py_path, build_dir / "main.py")

    # Step 6: Build with PyInstaller (mode + platform specific)
    build_pyinstaller_bundle(
        venv_python=venv_python,
        main_path=build_dir / "main.py",
        dist_dir=dist_dir,
        build_dir=build_dir,
        mode=mode,
        target_platform=target_platform
    )

    # Step 6.1: For dev mode, we excluded app and plugins from PyInstaller analysis
    # so we must manually copy them to the dist directory
    if mode == "dev":
        print(f"Manually copying excluded source folders to {app_dist_dir}")
        copy_source_files(root_dir / "app", app_dist_dir / "app", config["exclude_patterns"], mode)
        copy_source_files(root_dir / "plugins", app_dist_dir / "plugins", config["exclude_patterns"], mode)

    # Step 7: Handle llama-cpp binaries (platform-specific)
    llama_cpp_dest = dist_dir / "llama-cpp"
    if llama_cpp_src.exists() and llama_cpp_src.resolve() == llama_cpp_dest.resolve():
        # If destination already exists (likely preserved during cleanup), skip copy if it's the same path
        print(f"✓ Using existing llama-cpp binaries in {dist_dir} (matching source)")
        
        # Fix rpaths (Linux/mac only)
        if target_platform in ["linux", "mac"]:
            llama_server = llama_cpp_dest / ("llama-server.exe" if target_platform == "win" else "llama-server")
            fix_llama_cpp_rpaths(llama_server)
    else:
        print(f"Warning: llama-cpp not found at {llama_cpp_src}; skipping (mode: {mode})", file=sys.stderr)

    # Step 8: Handle whisper-cpp binaries (platform-specific)
    whisper_cpp_dest = dist_dir / "whisper-cpp"
    if whisper_cpp_src.exists() and whisper_cpp_src.resolve() == whisper_cpp_dest.resolve():
        # If destination already exists (likely preserved during cleanup), skip copy if it's the same path
        print(f"✓ Using existing whisper-cpp binaries in {dist_dir} (matching source)")
        
        # Fix rpaths (Linux/mac only)
        if target_platform in ["linux", "mac"]:
            whisper_server = whisper_cpp_dest / ("whisper-server.exe" if target_platform == "win" else "whisper-server")
            fix_llama_cpp_rpaths(whisper_server)
    else:
        print(f"Warning: whisper-cpp not found at {whisper_cpp_src}; skipping (mode: {mode})", file=sys.stderr)

    # Step 9: Write README (platform + mode specific)
    write_readme(dist_dir, python_version, mode, target_platform)

    # Step 10: Cleanup build directory (mode-specific)
    if config["clean_build"]:
        print(f"Cleaning up {mode} build artifacts for {target_platform}")
        # Keep venv (unless no_cache is True)
        if no_cache:
            shutil.rmtree(build_dir / f"venv", ignore_errors=True)

    # Step 11: Compress the distribution directory to a ZIP file
    if no_packing:
        print(f"\n✓ Successfully built services in {dist_dir} (mode: {mode}, target: {target_platform})")
        print("   Skipping ZIP packing as requested.")
        return

    zip_filename = f"local-cocoa-server-{target_platform}-{mode}.zip"
    zip_path = output_dir.resolve() / zip_filename
    zip_directory(dist_dir, zip_path)

    # Final output
    print(f"\n✓ Successfully packaged services into {zip_path} (mode: {mode}, target: {target_platform})")
    print(f"   Using Python {python_version}")
    print(f"\n{mode.upper()} Distribution ZIP contents:")
    # List the first 20 items in the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zip_contents = zipf.namelist()
        for i, item in enumerate(zip_contents):
            if i >= 20:
                print(f"  - ... (total {len(zip_contents)} items)")
                break
            print(f"  - {item}")

# --------------------------
# Argument Parsing & Entry Point
# --------------------------
def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Build Local cocoa service with PyInstaller")
    
    # Required arguments
    parser.add_argument(
        "--root_dir",
        required=True,
        type=Path,
        help="Root directory of the project (e.g., ../)"
    )
    parser.add_argument(
        "--platform",
        required=True,
        type=str,
        choices=["win", "linux", "mac"],
        help="Target build platform (win/linux/mac)"
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["dev", "prod"],
        help="Build mode (dev/prod)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory for distribution (e.g., ./dist)"
    )
    # no-cache should be used for prod build
    parser.add_argument(
        "--no-cache",
        action="store_true",  # Existence of the argument sets it to True, default to False
        help="Force recreate virtual environment and reinstall dependencies (default: False)"
    )
    parser.add_argument(
        "--no-packing",
        action="store_true",
        help="Skip ZIP packing operation (default: False)"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="If a previous build exists, only update source code. Faster for development. (default: False)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Validate and resolve paths
        root_dir = args.root_dir.resolve()
        output_dir = args.output_dir.resolve()
        build_mode: BuildMode = args.mode  # type: ignore
        target_platform: SupportedPlatforms = args.platform  # type: ignore
        no_cache: bool = args.no_cache
        no_packing: bool = args.no_packing
        incremental: bool = args.incremental
        
        # Run main build function
        main(root_dir, build_mode, target_platform, output_dir, no_cache, no_packing, incremental)
        sys.exit(0)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)