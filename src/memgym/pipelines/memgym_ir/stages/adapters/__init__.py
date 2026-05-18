"""Dataset adapters for MemGym-IR pipeline."""

from .musique import MuSiQueAdapter
from .hotpotqa import HotpotQAAdapter
from .synthetic import SyntheticAdapter
from .wikimultihop import WikiMultihopAdapter

ADAPTERS = {
    "musique": MuSiQueAdapter,
    "hotpotqa": HotpotQAAdapter,
    "synthetic": SyntheticAdapter,
    "2wikimultihop": WikiMultihopAdapter,
}
