"""MuSiQue dataset adapter for MemGym-IR pipeline."""

from typing import List, Dict, Any, Optional

from datasets import load_dataset

from ...types.schemas import FilteredIRInstance


class MuSiQueAdapter:
    """Load and parse MuSiQue dataset from HuggingFace."""

    name = "musique"

    def load_raw(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MuSiQue dataset from HuggingFace."""
        print("Loading MuSiQue dataset from HuggingFace...")
        dataset = load_dataset("bdsaglam/musique", split="train")
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        instances = list(dataset)
        print(f"Loaded {len(instances)} raw instances")
        return instances

    def parse(self, raw: Dict[str, Any], **kwargs) -> Optional[FilteredIRInstance]:
        """Parse a single MuSiQue instance into FilteredIRInstance."""
        instance_id = raw.get("id", "")
        question = raw.get("question", "")
        answer = raw.get("answer", "")
        answerable = raw.get("answerable", True)
        if isinstance(answerable, str):
            answerable = answerable.lower() == "true"

        # Parse paragraphs
        paragraphs = raw.get("paragraphs", [])
        supporting = []
        distractors = []
        para_by_idx = {}
        for p in paragraphs:
            is_sup = p.get("is_supporting", False)
            if isinstance(is_sup, str):
                is_sup = is_sup.lower() == "true"
            entry = {
                "title": p.get("title", ""),
                "text": p.get("paragraph_text", ""),
                "is_supporting": is_sup,
            }
            idx = p.get("idx", "")
            if isinstance(idx, int):
                idx = str(idx)
            para_by_idx[idx] = entry
            if is_sup:
                supporting.append(entry)
            else:
                distractors.append(entry)

        # Parse decomposition
        decomposition_raw = raw.get("question_decomposition", [])
        decomposition = []
        for i, d in enumerate(decomposition_raw):
            sub_q = d.get("question", "")
            sub_a = d.get("answer", "")
            para_support_idx = d.get("paragraph_support_idx")
            para_title = ""
            para_text = ""
            if para_support_idx is not None:
                idx_key = str(para_support_idx)
                if idx_key in para_by_idx:
                    para_title = para_by_idx[idx_key]["title"]
                    para_text = para_by_idx[idx_key]["text"]
                else:
                    for p in paragraphs:
                        if p.get("title", "") == str(para_support_idx):
                            para_title = p.get("title", "")
                            para_text = p.get("paragraph_text", "")
                            break

            decomposition.append({
                "sub_question": sub_q,
                "sub_answer": sub_a,
                "paragraph_title": para_title,
                "paragraph_text": para_text,
                "hop_index": i,
            })

        num_hops = len(decomposition)

        return FilteredIRInstance(
            instance_id=f"musique__{instance_id}",
            source_dataset="musique",
            question=question,
            answer=answer,
            num_hops=num_hops,
            decomposition=decomposition,
            supporting_paragraphs=supporting,
            distractor_paragraphs=distractors,
            question_type="bridge",
            answerable=answerable,
        )
