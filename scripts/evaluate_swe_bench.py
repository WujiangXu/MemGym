"""Moved to src/memgym/gym/swe_bench/evaluate.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from memgym.gym.swe_bench.evaluate import main
if __name__ == "__main__":
    main()
