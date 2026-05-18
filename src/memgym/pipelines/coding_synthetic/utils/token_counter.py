"""Token counting utilities using tiktoken."""

try:
    import tiktoken
    _ENC = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(text: str) -> int:
        """Count tokens using GPT-4 tiktoken encoding."""
        if not text:
            return 0
        return len(_ENC.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        """Fallback: approximate tokens as len // 4."""
        if not text:
            return 0
        return len(text) // 4
