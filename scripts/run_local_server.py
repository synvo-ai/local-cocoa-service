import os
import subprocess
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------
# Cross-platform configuration (Windows/Mac)
# --------------------------
# Get current operating system
os_name = platform.system()

# Set executable suffix and shell type based on OS
if os_name == "Windows":
    exe_suffix = ".exe"
    use_shell = True  # Windows requires shell=True for exe execution in some cases
elif os_name == "Darwin":  # Darwin represents macOS
    exe_suffix = ""  # Mac executable has no .exe suffix
    use_shell = False  # Mac does not need shell=True for binary execution
else:
    raise RuntimeError(f"Unsupported operating system: {os_name} (only Windows/Mac are supported)")

# --------------------------
# Environment variables setup (effective only in current process)
# --------------------------
os.environ["DEBUG"] = "true"
os.environ["LOCAL_ACTIVE_AUDIO_MODEL_ID"] = "whisper-small"
os.environ["LOCAL_ACTIVE_EMBEDDING_MODEL_ID"] = "embedding-q4"
os.environ["LOCAL_ACTIVE_MODEL_ID"] = "vlm"
os.environ["LOCAL_ACTIVE_RERANKER_MODEL_ID"] = "reranker"

# Relative paths (based on D:\Workspace\Synvo\local-cocoa\local-cocoa-service)
os.environ["LOCAL_MODEL_ROOT_PATH"] = r"runtime/local-cocoa-models/pretrained/"
os.environ["LOCAL_MODELS_CONFIG_PATH"] = r"models.config.json"
os.environ["LOCAL_RUNTIME_ROOT"] = r"runtime"
os.environ["LOCAL_SERVICE_LOG_TO_FILE"] = "true"

# --------------------------
# Executable path configuration (cross-platform)
# --------------------------

# Assemble executable path with OS-specific suffix
exe_name = f"local-cocoa-server{exe_suffix}"
exe_path = os.path.join(PROJECT_ROOT, "dist", "local-cocoa-server", exe_name)  # Use os.path.join for cross-platform path splicing

# --------------------------
# Execute the server executable
# --------------------------
try:
    print(f"Starting {exe_path} (OS: {os_name})...")
    # Execute with OS-specific shell configuration
    subprocess.run(
        exe_path,
        shell=use_shell,
        check=True,
        cwd=PROJECT_ROOT,         # Ensure execution in the project root
    )
except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Failed to execute {exe_path}: {e}") from e
except FileNotFoundError:
    raise FileNotFoundError(
        f"Executable file not found: {exe_path}\n"
        f"Please check if the file exists, and confirm the suffix is correct for {os_name}."
    ) from None
finally:
    # Pause to prevent window closing (simulate BAT's pause)
    if os_name == "Windows":
        input("Press Enter to exit...")  # Windows: interactive pause
    else:
        # Mac: wait for user input (consistent with Windows logic)
        input("Press Enter to exit...")