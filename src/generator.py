import re, torch
from llama_cpp import Llama

class Qwen3Generator:
    
    SYSTEM_PROMPT = """
    You are TurboQwenRAG, a helpful AI assistant that answers questions based ONLY on the provided context documents.

    Rules:
    - Do NOT make up information.
    - Answer ONLY from the given context. Do not use outside knowledge.
    - Be concise, accurate, and clear.
    - Mention which document the info is from when possible.
    - Respond in the same language as the question.
    """

    def __init__(
        self,
        model_path   : str = "./models/Qwen3-8B-Q4_K_M.gguf",
        n_ctx        : int = 8192,
        n_gpu_layers : int = -1,
    ):
        print(f"Loading Qwen3-8B from {model_path}...")
        self.llm = Llama(
            model_path   = model_path,
            n_gpu_layers = n_gpu_layers,
            n_ctx        = n_ctx,
            n_batch      = 512,
            verbose      = False,
            chat_format  = "chatml",
        )
        print("Qwen3-8B loaded!")

        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            used, total = result.stdout.strip().split(", ")
            print(f"   VRAM used: {int(used)/1024:.1f} GB / {int(total)/1024:.1f} GB")
        except Exception:
            pass

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Delete <think>...</think> tag from Qwen3 output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(
        self,
        question   : str,
        context    : str,
        max_tokens : int   = 512,
        temperature: float = 0.1,
    ) -> str:
        user_message = f"""Context Documents: {context}
        ---

        Question: {question} /no_think"""

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens     = max_tokens,
            temperature    = temperature,
            top_p          = 0.9,
            repeat_penalty = 1.1,
        )
        raw    = response["choices"][0]["message"]["content"].strip()
        answer = self._strip_thinking(raw)  # strip if there is a thinking tag
        return answer