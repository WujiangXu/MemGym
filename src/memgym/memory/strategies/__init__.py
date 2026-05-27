"""Generic, benchmark-agnostic memory strategies.

Each module here registers one strategy by name via ``register_memory_model``;
the package root (``memgym.memory``) imports them to populate the registry.
Strategies are resolved by name, never by import path, so files may move freely
within this subpackage without affecting the public ``--memory`` flag.
"""
