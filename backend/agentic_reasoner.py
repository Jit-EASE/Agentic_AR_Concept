import os
from typing import Optional

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class AgenticReasoner:
    """
    Uses GPT-4o-mini (or any text model) to generate a short, functional description
    of the detected object, with an emphasis on Irish / farm / daily context.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if _HAS_OPENAI:
            self.client = OpenAI()
        else:
            self.client = None

    def explain(self, label: str) -> Optional[str]:
        if not self.enabled or self.client is None:
            return None

        prompt = (
            f"You are an expert in Irish daily life, agriculture and built environment. "
            f"In one short sentence (max 25 words), explain what a '{label}' is and what "
            f"its main function is, speaking in neutral, professional English."
        )

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise technical explainer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=80,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            return None
