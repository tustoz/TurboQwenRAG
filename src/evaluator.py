import re
import numpy as np


class RAGEvaluator:
    def __init__(self, generator, embedder):
        self.generator = generator
        self.embedder = embedder

    def _ask(self, prompt: str) -> str:
        response = self.generator.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation assistant. "
                        "Be concise and precise. Follow the output format exactly."
                    ),
                },
                {"role": "user", "content": prompt + " /no_think"},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        raw = response["choices"][0]["message"]["content"].strip()
        return self.generator._strip_thinking(raw)

    def context_relevance(self, question: str, contexts: list) -> float:
        q_emb = self.embedder.embed_query(question)
        scores = []
        for ctx in contexts:
            c_emb = self.embedder.embed_documents([ctx])[0]
            scores.append(float(np.dot(q_emb, c_emb)))
        return float(np.mean(scores)) if scores else 0.0

    def faithfulness(self, answer: str, contexts: list) -> float:
        context_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            "Given the following context documents and an AI-generated answer, "
            "extract all factual claims from the answer. "
            "For each claim, check whether it is supported by the context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Answer:\n{answer}\n\n"
            "List each claim on its own line starting with SUPPORTED or NOT_SUPPORTED.\n"
            "Example:\nSUPPORTED: The sky is blue.\nNOT_SUPPORTED: Grass is red.\n\n"
            "Evaluate now:"
        )
        result = self._ask(prompt)
        supported = len(re.findall(r"^SUPPORTED:", result, re.MULTILINE))
        not_supported = len(re.findall(r"^NOT_SUPPORTED:", result, re.MULTILINE))
        total = supported + not_supported
        return supported / total if total > 0 else 1.0

    def answer_relevance(self, question: str, answer: str) -> float:
        prompt = (
            "Given this answer, generate 3 questions that this answer could be responding to. "
            "One question per line, numbered 1. 2. 3.\n\n"
            f"Answer: {answer}\n\nQuestions:"
        )
        result = self._ask(prompt)
        lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
        questions = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if len(cleaned) > 5:
                questions.append(cleaned)
        if not questions:
            return 0.0
        q_emb = self.embedder.embed_query(question)
        gen_embs = self.embedder.embed_documents(questions)
        scores = [float(np.dot(q_emb, e)) for e in gen_embs]
        return float(np.mean(scores))

    def evaluate(self, question: str, answer: str, contexts: list) -> dict:
        ctx_rel = self.context_relevance(question, contexts)
        faith = self.faithfulness(answer, contexts)
        ans_rel = self.answer_relevance(question, answer)
        overall = (ctx_rel + faith + ans_rel) / 3
        return {
            "context_relevance": round(ctx_rel, 4),
            "faithfulness": round(faith, 4),
            "answer_relevance": round(ans_rel, 4),
            "overall_score": round(overall, 4),
        }
