"""2WikiMultihopQA dataset adapter for MemGym-IR pipeline.

2WikiMultihopQA provides multi-hop questions with 2-5 hops, explicit
decompositions (evidences), and supporting/distractor paragraphs.
Key advantage: native 3-5 hop instances without upgrade needed.

Dataset: https://huggingface.co/datasets/framolfese/2WikiMultihopQA
"""

from typing import List, Dict, Any, Optional

from datasets import load_dataset

from ...types.schemas import FilteredIRInstance


class WikiMultihopAdapter:
    """Load and parse 2WikiMultihopQA dataset from HuggingFace."""

    name = "2wikimultihop"

    def load_raw(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load 2WikiMultihopQA dataset from HuggingFace."""
        print("Loading 2WikiMultihopQA dataset from HuggingFace...")
        try:
            dataset = load_dataset(
                "framolfese/2WikiMultihopQA",
                split="train",
            )
        except Exception:
            # Fallback: try original (may require dataset scripts support)
            try:
                dataset = load_dataset("xanhho/2WikiMultihopQA", split="train")
            except Exception:
                print("Warning: Could not load 2WikiMultihopQA. "
                      "Try: pip install datasets && huggingface-cli login")
                return []

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        instances = list(dataset)
        print(f"Loaded {len(instances)} raw instances")
        return instances

    def parse(self, raw: Dict[str, Any], **kwargs) -> Optional[FilteredIRInstance]:
        """Parse a single 2WikiMultihopQA instance into FilteredIRInstance.

        2WikiMultihopQA schema:
        - _id: instance ID
        - question: the question
        - answer: gold answer
        - type: reasoning type (bridge, comparison, inference, compositional)
        - context: list of [title, [sentences...]] pairs
        - supporting_facts: list of [title, sent_idx] pairs
        - evidences: list of [subject, relation, object, ...] tuples
        """
        instance_id = raw.get("_id", raw.get("id", ""))
        question = raw.get("question", "")
        answer = raw.get("answer", "")
        q_type = raw.get("type", "bridge")

        # Parse context paragraphs
        context = raw.get("context", [])
        supporting_titles = set()
        sup_facts = raw.get("supporting_facts", [])

        # supporting_facts can be list of [title, sent_idx] or dict with title/sent_id
        if isinstance(sup_facts, dict):
            titles_list = sup_facts.get("title", [])
            supporting_titles = set(titles_list)
        elif isinstance(sup_facts, list):
            for sf in sup_facts:
                if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                    supporting_titles.add(sf[0])
                elif isinstance(sf, dict):
                    supporting_titles.add(sf.get("title", ""))

        supporting = []
        distractors = []

        # context can be list of [title, sentences] or dict
        if isinstance(context, dict):
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                is_sup = title in supporting_titles
                entry = {"title": title, "text": text, "is_supporting": is_sup}
                if is_sup:
                    supporting.append(entry)
                else:
                    distractors.append(entry)
        elif isinstance(context, list):
            for item in context:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    title = item[0]
                    sentences = item[1]
                    text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                else:
                    continue
                is_sup = title in supporting_titles
                entry = {"title": title, "text": text, "is_supporting": is_sup}
                if is_sup:
                    supporting.append(entry)
                else:
                    distractors.append(entry)

        if len(supporting) < 2:
            return None

        # Build decomposition from evidences
        evidences = raw.get("evidences", [])
        decomposition = self._build_decomposition(
            question, answer, supporting, evidences,
        )

        if not decomposition:
            return None

        num_hops = len(decomposition)

        return FilteredIRInstance(
            instance_id=f"2wiki__{instance_id}",
            source_dataset="2wikimultihop",
            question=question,
            answer=answer,
            num_hops=num_hops,
            decomposition=decomposition,
            supporting_paragraphs=supporting,
            distractor_paragraphs=distractors,
            question_type=q_type,
            answerable=True,
        )

    def _build_decomposition(
        self,
        question: str,
        answer: str,
        supporting: List[Dict],
        evidences: List,
    ) -> List[Dict]:
        """Build decomposition from evidences and supporting paragraphs.

        Evidences are typically triples like [subject, relation, object]
        that define each reasoning hop.
        """
        decomposition = []

        if evidences:
            # Use evidences to build decomposition
            for i, evidence in enumerate(evidences):
                if isinstance(evidence, (list, tuple)) and len(evidence) >= 3:
                    subject = evidence[0]
                    relation = evidence[1]
                    obj = evidence[2]
                    sub_question = f"What is the {relation} of {subject}?"
                    sub_answer = str(obj)

                    # Find matching supporting paragraph
                    para_title = ""
                    para_text = ""
                    for p in supporting:
                        if subject.lower() in p["title"].lower() or \
                           subject.lower() in p["text"].lower():
                            para_title = p["title"]
                            para_text = p["text"]
                            break

                    decomposition.append({
                        "sub_question": sub_question,
                        "sub_answer": sub_answer,
                        "paragraph_title": para_title,
                        "paragraph_text": para_text,
                        "hop_index": i,
                    })
                elif isinstance(evidence, dict):
                    subject = evidence.get("subject", "")
                    relation = evidence.get("relation", "")
                    obj = evidence.get("object", "")
                    sub_question = f"What is the {relation} of {subject}?"
                    sub_answer = str(obj)

                    para_title = ""
                    para_text = ""
                    for p in supporting:
                        if subject.lower() in p["title"].lower() or \
                           subject.lower() in p["text"].lower():
                            para_title = p["title"]
                            para_text = p["text"]
                            break

                    decomposition.append({
                        "sub_question": sub_question,
                        "sub_answer": sub_answer,
                        "paragraph_title": para_title,
                        "paragraph_text": para_text,
                        "hop_index": i,
                    })

        if not decomposition:
            # Fallback: one hop per supporting paragraph
            for i, para in enumerate(supporting):
                is_last = i == len(supporting) - 1
                decomposition.append({
                    "sub_question": f"Sub-question {i+1} for: {question}",
                    "sub_answer": answer if is_last else para["title"],
                    "paragraph_title": para["title"],
                    "paragraph_text": para["text"],
                    "hop_index": i,
                })

        return decomposition
