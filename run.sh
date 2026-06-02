#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────
# run.sh — Activate environment & run sample
# ─────────────────────────────────────────
#
# Usage:
#   ./run.sh                # run the Z-Image-Turbo sample (default)
#   ./run.sh zimage         # same
#   ./run.sh gen-prompt     # run the prompt generation sample
#   ./run.sh all            # run both samples
#
# Environment variables (for gen-prompt.py):
#   NVIDIA_BASE_URL    – NVIDIA API base URL
#   NVIDIA_API_KEY     – NVIDIA API key
#   NVIDIA_MODEL       – NVIDIA model name

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="ai312"
CONDA_BASE="${CONDA_BASE:-$HOME/Applications/miniforge3}"

SAMPLE="${1:-zimage}"

# --- helpers -------------------------------------------------
red()  { printf '\033[31m%s\033[0m\n' "$*"; }
green(){ printf '\033[32m%s\033[0m\n' "$*"; }
bold() { printf '\033[1m%s\033[0m\n' "$*"; }

# --- activate conda ------------------------------------------
activate_env() {
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
    # shellcheck source=/dev/null
    if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV"
    else
      red "ERROR: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh"
      red "Set CONDA_BASE to your miniforge/anaconda install path."
      exit 1
    fi
  fi
  green "✓ Conda environment '$CONDA_ENV' active  (python $(python --version 2>&1 | awk '{print $2}'))"
}

# --- install deps if needed ----------------------------------
install_deps() {
  local missing=false
  for pkg in torch diffusers openai accelerate transformers; do
    if ! python -c "import ${pkg//-/_}" 2>/dev/null; then
      red "✗ Missing package: $pkg"
      missing=true
    fi
  done
  if $missing; then
    bold "Installing missing packages…"
    pip install -r "$PROJECT_DIR/requirements.txt"
  fi
  green "✓ All dependencies available"
}

# --- run z-image sample --------------------------------------
run_zimage() {
  bold "═══ Running Z-Image-Turbo sample ═══"
  cd "$PROJECT_DIR"
  python "$PROJECT_DIR/z-image.py"
  if [[ -f "$PROJECT_DIR/output/zimage.png" ]]; then
    open "$PROJECT_DIR/output/zimage.png"
  fi
}

# --- run gen-prompt sample -----------------------------------
run_gen_prompt() {
  bold "═══ Running Prompt Generation sample ═══"
  local missing_vars=()
  for var in NVIDIA_BASE_URL NVIDIA_API_KEY NVIDIA_MODEL; do
    if [[ -z "${!var:-}" ]]; then
      missing_vars+=("$var")
    fi
  done
  if [[ ${#missing_vars[@]} -gt 0 ]]; then
    red "ERROR: Missing environment variables: ${missing_vars[*]}"
    red "Set them before running this sample, e.g.:"
    red "  export NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1"
    red "  export NVIDIA_API_KEY=nvapi-..."
    red "  export NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct"
    exit 1
  fi
  cd "$PROJECT_DIR"
  python "$PROJECT_DIR/gen-prompt.py"
}

# --- main ----------------------------------------------------
activate_env
install_deps

case "$SAMPLE" in
  zimage)
    run_zimage
    ;;
  gen-prompt)
    run_gen_prompt
    ;;
  all)
    run_gen_prompt
    echo
    run_zimage
    ;;
  *)
    red "Unknown sample: '$SAMPLE'"
    echo "Usage: ./run.sh [zimage|gen-prompt|all]"
    exit 1
    ;;
esac

green "✓ Done."
