from typing import Optional, List
import httpx
from app.ports.llm import LLMProviderPort
from app.config import settings

class OpenAICompatLLM(LLMProviderPort):
    def __init__(self, base_url: Optional[str] = 'http://localhost:11434/v1', api_key: Optional[str] = None, model: Optional[str] = 'deepseek-r1:8b', timeout: float = 60.0):
        self.base_url = base_url or settings.LLM_OPENAI_BASE
        self.api_key = api_key or settings.LLM_OPENAI_API_KEY
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout

    async def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful financial analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]