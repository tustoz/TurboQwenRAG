import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Qwen3Reranker:
    MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
    INSTRUCTION = "Given a question, retrieve relevant passages that answer the question"
    PREFIX = (
        "<|im_start|>system\n"
        'Judge whether the Document meets the requirements based on the Query and the '
        'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, device: str = "cuda", max_length: int = 512):
        print("Loading Qwen3-Reranker-0.6B...")
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        self.model.eval()
        self.token_true_id = self.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
        self.token_false_id = self.tokenizer("no", add_special_tokens=False)["input_ids"][0]
        print("Qwen3-Reranker-0.6B ready!")

    def _format(self, query: str, doc: str) -> str:
        pair = (
            f"<Instruct>: {self.INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )
        return self.PREFIX + pair + self.SUFFIX

    def _score(self, query: str, doc: str) -> float:
        text = self._format(query, doc)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        true_score = logits[self.token_true_id].item()
        false_score = logits[self.token_false_id].item()
        score = torch.softmax(torch.tensor([true_score, false_score]), dim=0)[0].item()
        return score

    def rerank(self, query: str, results: list, top_k: int = None) -> list:
        for r in results:
            r["rerank_score"] = self._score(query, r["content"])
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results