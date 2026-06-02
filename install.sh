#!/bin/bash
# MemGym Installation Script using UV
# Works with conda, venv, or system Python

set -e  # Exit on error

echo "=================================="
echo "MemGym Installation with UV"
echo "=================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Resolve the interpreter to use for verification. Prefer a local venv
# (`.venv/bin/python`), then `python3`, then `python`. Some hosts have
# only `python3` on PATH, so plain `python` would fail the verification
# footer even though `uv pip install` succeeded.
if [ -x ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    VENV_PY="python3"
elif command -v python &> /dev/null; then
    VENV_PY="python"
else
    echo -e "${RED}No python interpreter found. Install Python 3.10+ first.${NC}"
    exit 1
fi
echo -e "${GREEN}Using Python: ${VENV_PY}${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}UV not found. Installing UV...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}UV installed successfully!${NC}"
fi

echo -e "\n${GREEN}UV version:${NC}"
uv --version

# Parse arguments
INSTALL_TAU2=false
INSTALL_SWE=false
INSTALL_OPENHANDS=false
INSTALL_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tau2)
            INSTALL_TAU2=true
            shift
            ;;
        --swe)
            INSTALL_SWE=true
            shift
            ;;
        --openhands)
            INSTALL_OPENHANDS=true
            shift
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./install.sh [--tau2] [--swe] [--openhands] [--all]"
            exit 1
            ;;
    esac
done

# If no args, install core only
if [ "$INSTALL_TAU2" = false ] && [ "$INSTALL_SWE" = false ] && [ "$INSTALL_OPENHANDS" = false ] && [ "$INSTALL_ALL" = false ]; then
    echo -e "\n${YELLOW}Installing MemGym core only${NC}"
    echo "Use --tau2, --swe, --openhands, or --all to install benchmarks"
fi

if [ "$INSTALL_ALL" = true ]; then
    INSTALL_TAU2=true
    INSTALL_SWE=true
    INSTALL_OPENHANDS=true
fi

# Install core MemGym requirements
echo -e "\n${GREEN}[1/X] Installing MemGym core requirements...${NC}"
uv pip install -r requirements.txt

MISSING_CLONES=()

# Install tau2-bench
if [ "$INSTALL_TAU2" = true ]; then
    echo -e "\n${GREEN}[2/X] Installing tau2-bench requirements...${NC}"
    uv pip install -r requirements-tau2.txt

    echo -e "\n${GREEN}[3/X] Installing tau2-bench package...${NC}"
    if [ -d "third_party/tau2-bench" ]; then
        cd third_party/tau2-bench
        uv pip install -e .
        cd ../..
        echo -e "${GREEN}✓ tau2-bench installed${NC}"
    else
        echo -e "${RED}✗ third_party/tau2-bench not found${NC}"
        echo "  Clone it manually with:"
        echo "    git clone https://github.com/sierra-research/tau2-bench.git third_party/tau2-bench"
        echo "  then re-run install.sh --tau2 (or --all)."
        MISSING_CLONES+=("tau2-bench")
    fi
fi

# Install SWE-bench
if [ "$INSTALL_SWE" = true ]; then
    echo -e "\n${GREEN}[4/X] Installing SWE-bench requirements...${NC}"
    uv pip install -r requirements-swe.txt

    echo -e "\n${GREEN}[5/X] Installing mini-swe-agent...${NC}"
    # Prefer third_party local clone, fall back to PyPI
    if [ -d "third_party/mini-swe-agent" ]; then
        cd third_party/mini-swe-agent
        uv pip install -e .
        cd ../..
        echo -e "${GREEN}✓ mini-swe-agent installed (from third_party/)${NC}"
    else
        echo -e "${YELLOW}third_party/mini-swe-agent not found, installing from PyPI...${NC}"
        uv pip install mini-swe-agent
        echo -e "${GREEN}✓ mini-swe-agent installed (from PyPI)${NC}"
    fi

    echo -e "\n${GREEN}[6/X] Installing SWE-bench harness...${NC}"
    # Prefer third_party local clone, fall back to PyPI
    if [ -d "third_party/SWE-bench" ]; then
        cd third_party/SWE-bench
        uv pip install -e .
        cd ../..
        echo -e "${GREEN}✓ swebench installed (from third_party/)${NC}"
    else
        echo -e "${YELLOW}third_party/SWE-bench not found, installing from PyPI...${NC}"
        uv pip install swebench
        echo -e "${GREEN}✓ swebench installed (from PyPI)${NC}"
    fi
fi

# Install OpenHands (CodeAct agent)
if [ "$INSTALL_OPENHANDS" = true ]; then
    echo -e "\n${GREEN}[7/X] Installing OpenHands (CodeAct agent)...${NC}"
    if [ -d "third_party/OpenHands" ]; then
        cd third_party/OpenHands
        uv pip install -e .
        cd ../..
        echo -e "${GREEN}✓ OpenHands installed${NC}"
    else
        echo -e "${RED}✗ third_party/OpenHands not found${NC}"
        echo "  Clone it manually with:"
        echo "    git clone https://github.com/All-Hands-AI/OpenHands.git third_party/OpenHands"
        echo "  then re-run install.sh --openhands (or --all)."
        echo "  Note: the MemGym CodeAct wrapper currently targets the legacy"
        echo "  openhands.core/events API and is incompatible with openhands-ai>=1.7.0."
        MISSING_CLONES+=("OpenHands")
    fi
fi

# Install MemGym itself
echo -e "\n${GREEN}[FINAL] Installing MemGym in editable mode...${NC}"
uv pip install -e .

# Verification
echo -e "\n${YELLOW}Verifying installation...${NC}"
"$VENV_PY" -c "
import sys
sys.path.insert(0, 'src')

try:
    from memgym.envs import list_envs
    from memgym.memory import list_memory_models
    from memgym.agents import list_reasoning_models
    print('✓ MemGym imports successful')
    print(f'  - Environments: {list_envs()}')
    print(f'  - Memory models: {list_memory_models()}')
    print(f'  - Reasoning models: {list_reasoning_models()}')
except Exception as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"

if [ "$INSTALL_TAU2" = true ] && [ -d "third_party/tau2-bench" ]; then
    "$VENV_PY" -c "
import sys
try:
    import tau2
    print('✓ tau2-bench available')
except ImportError:
    print('✗ tau2-bench not available')
    sys.exit(1)
"
fi

if [ "$INSTALL_SWE" = true ]; then
    "$VENV_PY" -c "
import sys
try:
    import minisweagent
    print('✓ mini-swe-agent available')
except ImportError:
    print('✗ mini-swe-agent not available')
    sys.exit(1)

try:
    import swebench
    print('✓ SWE-bench available (optional)')
except ImportError:
    print('⚠ SWE-bench not available (optional)')
"
fi

if [ "$INSTALL_OPENHANDS" = true ] && [ -d "third_party/OpenHands" ]; then
    "$VENV_PY" -c "
import sys
sys.path.insert(0, 'src')
try:
    import openhands
except ImportError:
    print('✗ OpenHands not available')
    sys.exit(1)
# Top-level 'import openhands' succeeding is NOT enough: openhands-ai>=1.7.0
# imports but drops the legacy openhands.core/events API the MemGym CodeAct
# wrapper needs. Verify the wrapper itself so the footer is honest.
from memgym.agents.codeact_wrapper import _OPENHANDS_AVAILABLE, _OPENHANDS_IMPORT_ERROR
if _OPENHANDS_AVAILABLE:
    print('✓ OpenHands available (CodeAct wrapper ready)')
else:
    print('⚠ OpenHands installed but the MemGym CodeAct wrapper is incompatible:')
    print('  ' + (_OPENHANDS_IMPORT_ERROR or 'unknown import error'))
    print('  --agent codeact is disabled; other tracks are unaffected.')
"
fi

# Honest exit: if --tau2/--openhands/--all was requested but the upstream
# clone is missing, the install is INCOMPLETE — fail loudly so CI catches it.
if [ ${#MISSING_CLONES[@]} -ne 0 ]; then
    echo -e "\n${RED}=================================="
    echo -e "✗ Installation INCOMPLETE"
    echo -e "==================================${NC}"
    echo -e "${RED}Missing third-party clones: ${MISSING_CLONES[*]}${NC}"
    echo "See the clone commands printed above. install.sh does NOT clone"
    echo "third-party repos for you — pre-clone them under third_party/ first."
    exit 1
fi

echo -e "\n${GREEN}=================================="
echo "✓ Installation Complete!"
echo "==================================${NC}"

echo -e "\n${GREEN}Next steps:${NC}"
echo "  1. Set API key: export OPENAI_API_KEY='your-key'"
echo "  2. Run tests: pytest tests/unit"
echo "  3. Run episode: python examples/run_episode.py --help"
