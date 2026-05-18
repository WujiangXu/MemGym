"""QA pipeline stages: convert, verify, eval, score."""

from .convert import convert_all, resolve_target_tokens, batch_extract_repo_files
from .verify import verify_all, dedup_qa_pairs
from .eval import evaluate_all, compare_strategies
from .score import format_report
