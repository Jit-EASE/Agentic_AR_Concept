from typing import Optional, Dict

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class AgenticReasoner:
    """
    Multi-agent reasoning layer for a detected object.

    Produces structured outputs for:
      - SUSTAINABILITY
      - SUPPLY_CHAIN
      - ECONOMETRICS
      - HAZARD
      - LPIS_GEO
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if _HAS_OPENAI:
            self.client = OpenAI()
        else:
            self.client = None

    def _call_openai(self, label: str) -> Optional[Dict[str, str]]:
        if not self.enabled or self.client is None:
            return None

        user_prompt = f"""
You are an expert in Irish agriculture, supply chains, sustainability and rural infrastructure.

Object detected: '{label}'.

For THIS object, give concise, practical, one-sentence insights for each agent below.
Stay within 18 words per line. Assume a modern Irish / EU farm context.

Return EXACTLY in this format (no extra commentary):

SUSTAINABILITY: <one sentence about emissions / resource use / environmental impact>
SUPPLY_CHAIN: <one sentence about its role in inputs/outputs, logistics, or processing>
ECONOMETRICS: <one sentence about how it affects yields, costs, volatility, or risk in models>
HAZARD: <one sentence about safety, legal or operational risks, especially on-farm>
LPIS_GEO: <one sentence about likely LPIS/CORINE class or spatial relevance in Ireland>
        """.strip()

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise technical explainer for Irish farm systems, "
                            "combining sustainability, supply chains, econometrics and risk."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=260,
            )
            text = completion.choices[0].message.content.strip()
        except Exception:
            return None

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        out: Dict[str, str] = {
            "SUSTAINABILITY": "",
            "SUPPLY_CHAIN": "",
            "ECONOMETRICS": "",
            "HAZARD": "",
            "LPIS_GEO": "",
        }

        for ln in lines:
            upper = ln.upper()
            for key in out.keys():
                prefix = key + ":"
                if upper.startswith(prefix):
                    value = ln[len(prefix):].strip(" :-")
                    out[key] = value
                    break

        return out

    def explain_structured(self, label: str) -> Optional[Dict[str, str]]:
        """
        Returns full multi-agent dict, suitable for HUD + side panel.
        """
        return self._call_openai(label)
