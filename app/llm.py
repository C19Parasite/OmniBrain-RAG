import os
from typing import Optional

class LLMManager:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return (
                "⚠️ [No API Key Found]\n\n"
                "Here is the retrieved context prepared for the LLM:\n\n"
                f"{prompt}"
            )

        try:
            from openai import OpenAI
            base_url = "https://api.groq.com/openai/v1" if "GROQ" in os.environ or "llama" in self.model_name else None
            client = OpenAI(api_key=self.api_key, base_url=base_url)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"❌ LLM Generation Error: {str(e)}"
