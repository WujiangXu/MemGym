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
        echo "Please clone tau2-bench into third_party/"
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
        echo "Run: git clone https://github.com/All-Hands-AI/OpenHands.git third_party/OpenHands"
    fi
fi

# Install MemGym itself
echo -e "\n${GREEN}[FINAL] Installing MemGym in editable mode...${NC}"
uv pip install -e .

echo -e "\n${GREEN}=================================="
echo "✓ Installation Complete!"
echo "==================================${NC}"

# Verification
echo -e "\n${YELLOW}Verifying installation...${NC}"
python -c "
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

if [ "$INSTALL_TAU2" = true ]; then
    python -c "
import sys
try:
    import tau2
    print('✓ tau2-bench available')
except ImportError:
    print('✗ tau2-bench not available')
    sys.exit(1)
" || true
fi

if [ "$INSTALL_SWE" = true ]; then
    python -c "
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
" || true
fi

if [ "$INSTALL_OPENHANDS" = true ]; then
    python -c "
import sys
try:
    import openhands
    print('✓ OpenHands available')
except ImportError:
    print('✗ OpenHands not available')
    sys.exit(1)
" || true
fi

echo -e "\n${GREEN}Next steps:${NC}"
echo "  1. Set API key: export OPENAI_API_KEY='your-key'"
echo "  2. Run tests: pytest tests/unit"
echo "  3. Run episode: python examples/run_episode.py --help"
