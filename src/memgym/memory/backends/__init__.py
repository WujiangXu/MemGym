"""Compression backends shared by multiple strategies.

These provide the low-level summarization / token-reduction machinery (LLM
summarizer calls, LLMLingua) that strategies in ``memgym.memory.strategies``
compose. They are not registered strategies themselves.
"""
