#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Local Cocoa CLI  –  bootstrap script
#
#  Activates the project virtual-env, ensures dependencies are
#  installed, and launches the interactive CLI.
#
#  Usage:
#    ./scripts/start_cli.sh              # interactive dashboard
#    ./scripts/start_cli.sh status       # one-shot status command
#    ./scripts/start_cli.sh mail list    # any subcommand
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve script directory → project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$PROJECT_ROOT/.venv"
REQUIREMENTS="$PROJECT_ROOT/app/requirements.txt"

# ── Create venv if missing ──────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
fi

# ── Activate ────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Install / update deps if requirements file is newer ─────
MARKER="$VENV_DIR/.deps-installed"
if [ ! -f "$MARKER" ] || [ "$REQUIREMENTS" -nt "$MARKER" ]; then
    echo "Installing dependencies …"
    pip install -q -r "$REQUIREMENTS"
    touch "$MARKER"
fi

# ── Launch CLI ──────────────────────────────────────────────
cd "$PROJECT_ROOT"
exec python -m cli "$@"
