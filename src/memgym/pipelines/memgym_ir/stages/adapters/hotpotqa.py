"""HotpotQA dataset adapter for MemGym-IR pipeline.

HotpotQA has no native decomposition. We use LLM (Haiku) to generate
sub-questions from the question + supporting paragraphs.
In dry-run mode, we create a heuristic decomposition from supporting facts.
"""

from typing import List, Dict, Any, Optional

from datasets import load_dataset

from ...types.schemas import FilteredIRInstance
from ...llm.client import LLMClient
from ...llm.prompts import HOTPOTQA_DECOMPOSITION_PROMPT, HOTPOTQA_DECOMPOSITION_SCHEMA


class HotpotQAAdapter:
    """Load and parse HotpotQA dataset from HuggingFace."""

    name = "hotpotqa"

    def load_raw(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load HotpotQA distractor setting from HuggingFace."""
        print("Loading HotpotQA dataset from HuggingFace...")
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        instances = list(dataset)
        print(f"Loaded {len(instances)} raw instances")
        return instances

    def parse(
        self,
        raw: Dict[str, Any],
        worker_client: Optional[LLMClient] = None,
        dry_run: bool = False,
        **kwargs,
    ) -> Optional[FilteredIRInstance]:
        """Parse a single HotpotQA instance into FilteredIRInstance.

        HotpotQA schema:
        - context.title: list of paragraph titles
        - context.sentences: list of list of sentences
        - supporting_facts.title: list of supporting paragraph titles
        - supporting_facts.sent_id: list of supporting sentence indices
        """
        instance_id = raw.get("id", "")
        question = raw.get("question", "")
        answer = raw.get("answer", "")
        q_type = raw.get("type", "bridge")

        # Build paragraphs from context
        titles = raw.get("context", {}).get("title", [])
        sentences_list = raw.get("context", {}).get("sentences", [])

        # Supporting fact titles
        sup_titles = set(raw.get("supporting_facts", {}).get("title", []))

        supporting = []
        distractors = []
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)
            is_sup = title in sup_titles
            entry = {
                "title": title,
                "text": text,
                "is_supporting": is_sup,
            }
            if is_sup:
                supporting.append(entry)
            else:
                distractors.append(entry)

        if len(supporting) < 2:
            return None

        # Generate decomposition
        if dry_run or worker_client is None:
            decomposition = self._heuristic_decomposition(
                question, answer, supporting
            )
        else:
            decomposition = self._llm_decomposition(
                question, answer, supporting, worker_client
            )

        if not decomposition:
            return None

        num_hops = len(decomposition)

        return FilteredIRInstance(
            instance_id=f"hotpotqa__{instance_id}",
            source_dataset="hotpotqa",
            question=question,
            answer=answer,
            num_hops=num_hops,
            decomposition=decomposition,
            supporting_paragraphs=supporting,
            distractor_paragraphs=distractors,
            question_type=q_type,
            answerable=True,
        )

    def _heuristic_decomposition(
        self,
        question: str,
        answer: str,
        supporting: List[Dict],
    ) -> List[Dict]:
        """Create a simple 2-hop decomposition from supporting paragraphs."""
        decomposition = []
        for i, para in enumerate(supporting[:2]):
            decomposition.append({
                "sub_question": f"Sub-question {i+1} for: {question}",
                "sub_answer": answer if i == len(supporting[:2]) - 1 else para["title"],
                "paragraph_title": para["title"],
                "paragraph_text": para["text"],
                "hop_index": i,
            })
        return decomposition

    def _llm_decomposition(
        self,
        question: str,
        answer: str,
        supporting: List[Dict],
        client: LLMClient,
    ) -> List[Dict]:
        """Use LLM to generate proper decomposition."""
        paras_str = ""
        for i, para in enumerate(supporting[:2]):
            paras_str += f"Paragraph {i+1}: [{para['title']}]\n{para['text']}\n\n"

        prompt = HOTPOTQA_DECOMPOSITION_PROMPT.format(
            question=question,
            answer=answer,
            paragraphs=paras_str,
        )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=HOTPOTQA_DECOMPOSITION_SCHEMA,
            temperature=0.3,
        )

        sub_questions = result.get("sub_questions", [])
        if not sub_questions:
            return self._heuristic_decomposition(question, answer, supporting)

        decomposition = []
        for i, sq in enumerate(sub_questions):
            decomposition.append({
                "sub_question": sq.get("sub_question", ""),
                "sub_answer": sq.get("sub_answer", ""),
                "paragraph_title": sq.get("paragraph_title", ""),
                "paragraph_text": next(
                    (p["text"] for p in supporting
                     if p["title"] == sq.get("paragraph_title", "")),
                    "",
                ),
                "hop_index": i,
            })
        return decomposition
