"""Synthetic LLM-generated dataset adapter for MemGym-IR pipeline.

Generates complete multi-hop QA instances from scratch using a 3-step LLM chain:
1. Generate related entities with factual paragraphs
2. Generate a multi-hop question + decomposition from those entities
3. Generate distractor paragraphs about similar entities
"""

import random
from typing import List, Dict, Any, Optional

from ...types.schemas import FilteredIRInstance
from ...llm.client import LLMClient
from ...llm.prompts import (
    SYNTHETIC_ENTITIES_PROMPT,
    SYNTHETIC_ENTITIES_SCHEMA,
    SYNTHETIC_QUESTION_PROMPT,
    SYNTHETIC_QUESTION_SCHEMA,
    SYNTHETIC_DISTRACTORS_PROMPT,
    SYNTHETIC_DISTRACTORS_SCHEMA,
)


DEFAULT_TOPICS = [
    "history", "geography", "science", "literature", "music",
    "sports", "politics", "technology", "art", "film",
    "medicine", "architecture", "economics", "philosophy", "astronomy",
    "biology", "chemistry", "mathematics", "linguistics", "anthropology",
]


class SyntheticAdapter:
    """Generate fully synthetic multi-hop QA instances via LLM."""

    name = "synthetic"

    def __init__(self, num_hops: int = 3, distractor_count: int = 5,
                 topic_list: Optional[List[str]] = None):
        self.num_hops = num_hops
        self.distractor_count = distractor_count
        self.topic_list = topic_list or DEFAULT_TOPICS

    def load_raw(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate placeholder 'raw' entries. Actual generation happens in parse()."""
        count = limit or 50
        print(f"Preparing {count} synthetic instance slots...")
        raw = []
        for i in range(count):
            topic = random.choice(self.topic_list)
            raw.append({
                "id": f"synthetic_{i:05d}",
                "topic": topic,
                "num_hops": self.num_hops,
                "distractor_count": self.distractor_count,
            })
        return raw

    def parse(
        self,
        raw: Dict[str, Any],
        worker_client: Optional[LLMClient] = None,
        dry_run: bool = False,
        **kwargs,
    ) -> Optional[FilteredIRInstance]:
        """Generate a complete instance via 3-step LLM chain."""
        instance_id = raw["id"]
        topic = raw.get("topic", "history")
        num_hops = raw.get("num_hops", self.num_hops)
        distractor_count = raw.get("distractor_count", self.distractor_count)

        if dry_run or worker_client is None:
            return self._dry_run_instance(instance_id, topic, num_hops)

        # Step 1: Generate entities + paragraphs
        entities = self._generate_entities(worker_client, topic, num_hops)
        if not entities or len(entities) < num_hops:
            return None

        # Step 2: Generate question + decomposition
        qa = self._generate_question(worker_client, entities, num_hops)
        if not qa or not qa.get("question") or not qa.get("decomposition"):
            return None

        # Step 3: Generate distractor paragraphs
        distractors = self._generate_distractors(
            worker_client, entities, distractor_count
        )

        # Assemble FilteredIRInstance
        supporting = [
            {"title": e["title"], "text": e["text"], "is_supporting": True}
            for e in entities[:num_hops]
        ]
        distractor_paras = [
            {"title": d["title"], "text": d["text"], "is_supporting": False}
            for d in distractors
        ]

        decomposition = []
        for i, step in enumerate(qa["decomposition"][:num_hops]):
            decomposition.append({
                "sub_question": step.get("sub_question", ""),
                "sub_answer": step.get("sub_answer", ""),
                "paragraph_title": step.get("paragraph_title", ""),
                "paragraph_text": next(
                    (e["text"] for e in entities
                     if e["title"] == step.get("paragraph_title", "")),
                    "",
                ),
                "hop_index": i,
            })

        return FilteredIRInstance(
            instance_id=f"synthetic__{instance_id}",
            source_dataset="synthetic",
            question=qa["question"],
            answer=qa.get("answer", ""),
            num_hops=len(decomposition),
            decomposition=decomposition,
            supporting_paragraphs=supporting,
            distractor_paragraphs=distractor_paras,
            question_type="bridge",
            answerable=True,
        )

    def _generate_entities(
        self, client: LLMClient, topic: str, num_hops: int,
    ) -> List[Dict]:
        """Step 1: Generate related entities with factual paragraphs."""
        prompt = SYNTHETIC_ENTITIES_PROMPT.format(
            topic=topic, num_entities=num_hops,
        )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=SYNTHETIC_ENTITIES_SCHEMA,
            temperature=0.8,
        )
        return result.get("entities", [])

    def _generate_question(
        self, client: LLMClient, entities: List[Dict], num_hops: int,
    ) -> Optional[Dict]:
        """Step 2: Generate multi-hop question + decomposition."""
        import json
        entities_json = json.dumps(entities, indent=2)
        prompt = SYNTHETIC_QUESTION_PROMPT.format(
            entities=entities_json, num_hops=num_hops,
        )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=SYNTHETIC_QUESTION_SCHEMA,
            temperature=0.7,
        )
        return result if result.get("question") else None

    def _generate_distractors(
        self, client: LLMClient, entities: List[Dict], count: int,
    ) -> List[Dict]:
        """Step 3: Generate distractor paragraphs."""
        entity_titles = ", ".join(e["title"] for e in entities)
        prompt = SYNTHETIC_DISTRACTORS_PROMPT.format(
            entity_titles=entity_titles, count=count,
        )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=SYNTHETIC_DISTRACTORS_SCHEMA,
            temperature=0.8,
        )
        return result.get("distractors", [])

    def _dry_run_instance(
        self, instance_id: str, topic: str, num_hops: int,
    ) -> FilteredIRInstance:
        """Create a placeholder instance for dry-run mode."""
        supporting = []
        decomposition = []
        for i in range(num_hops):
            title = f"Synthetic Entity {i+1} ({topic})"
            text = f"This is a synthetic paragraph about entity {i+1} in the domain of {topic}."
            supporting.append({"title": title, "text": text, "is_supporting": True})
            decomposition.append({
                "sub_question": f"What is entity {i+1}?",
                "sub_answer": f"Entity {i+1} answer",
                "paragraph_title": title,
                "paragraph_text": text,
                "hop_index": i,
            })

        return FilteredIRInstance(
            instance_id=f"synthetic__{instance_id}",
            source_dataset="synthetic",
            question=f"Synthetic {num_hops}-hop question about {topic}",
            answer=f"Synthetic answer about {topic}",
            num_hops=num_hops,
            decomposition=decomposition,
            supporting_paragraphs=supporting,
            distractor_paragraphs=[],
            question_type="bridge",
            answerable=True,
        )
