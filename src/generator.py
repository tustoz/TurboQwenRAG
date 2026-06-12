import re
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
        self.llm = Llama(
            model_path   = model_path,
            n_gpu_layers = n_gpu_layers,
            n_ctx        = n_ctx,
            n_batch      = 512,
            verbose      = False,
            chat_format  = "chatml",
        )

    def _build_messages(self, question: str, context: str, history: list = None) -> list:
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user",      "content": str(user_msg)})
                messages.append({"role": "assistant", "content": str(assistant_msg)})
        messages.append({"role": "user", "content": (
            f"Context Documents: {context}\n---\n\nQuestion: {question} /no_think"
        )})
        return messages

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(
        self,
        question   : str,
        context    : str,
        history    : list  = None,
        max_tokens : int   = 512,
        temperature: float = 0.1,
    ) -> str:
        response = self.llm.create_chat_completion(
            messages       = self._build_messages(question, context, history),
            max_tokens     = max_tokens,
            temperature    = temperature,
            top_p          = 0.9,
            repeat_penalty = 1.1,
        )
        raw = response["choices"][0]["message"]["content"].strip()
        return self._strip_thinking(raw)

    def generate_stream(
        self,
        question   : str,
        context    : str,
        history    : list  = None,
        max_tokens : int   = 512,
        temperature: float = 0.1,
    ):
        stream = self.llm.create_chat_completion(
            messages       = self._build_messages(question, context, history),
            max_tokens     = max_tokens,
            temperature    = temperature,
            top_p          = 0.9,
            repeat_penalty = 1.1,
            stream         = True,
        )
        in_think = False
        started  = False
        buf      = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" not in delta or not delta["content"]:
                continue
            buf += delta["content"]
            while buf:
                if in_think:
                    end = buf.find("</think>")
                    if end != -1:
                        buf      = buf[end + 8:]
                        in_think = False
                    else:
                        buf = ""
                else:
                    start = buf.find("<think>")
                    if start != -1:
                        if start > 0:
                            text = buf[:start] if started else buf[:start].lstrip()
                            if text:
                                started = True
                                yield text
                        buf      = buf[start + 7:]
                        in_think = True
                    else:
                        text = buf if started else buf.lstrip()
                        buf  = ""
                        if text:
                            started = True
                            yield text