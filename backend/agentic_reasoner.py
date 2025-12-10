import os
from typing import Optional, Dict

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class AgenticReasoner:
    """
    Multi-agent reasoning layer for a detected object.

    For each object label (e.g. 'tractor', 'silo', 'cow'), it calls GPT-4o-mini
    once with a structured prompt and derives 5 agent views:

      🌱 Sustainability agent
      📦 Supply chain agent
      🧮 Econometric forecasting agent
      🔍 Hazard detection agent
      📡 Geo-tag + LPIS match agent

    Returned value is a SHORT fused caption suitable for overlay.
    (You can later return the full dict and visualise it in a Spectre panel.)
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if _HAS_OPENAI:
            self.client = OpenAI()
        else:
            self.client = None

    def _call_openai(self, label: str) -> Optional[Dict[str, str]]:
        """
        Single GPT-4o-mini call that returns a multi-line, agent-structured answer.
        We then parse it into a dict.
        """
        if not self.enabled or self.client is None:
            return None

        user_prompt = f"""
You are a specialist in Irish agriculture, supply chains, and rural infrastructure.

Object detected: '{label}'.

For THIS object, give concise, **practical**, one-sentence insights for each agent below.
Stay within 15 words per line. Assume Irish / EU farm context.

Return EXACTLY in this format (no extra commentary):

SUSTAINABILITY: <one sentence about emissions / resource use / environmental impact>
SUPPLY_CHAIN: <one sentence about its role in inputs/outputs, logistics, or processing>
ECONOMETRICS: <one sentence about how it affects yields, costs, or risk in models>
HAZARD: <one sentence about safety, legal or operational risks>
LPIS_GEO: <one sentence about likely LPIS/CORINE class or spatial relevance in Ireland>
        """.strip()

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise, technical explainer of farm objects, "
                            "sustainability, supply chains and risk in an Irish context."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=220,
            )
            text = completion.choices[0].message.content.strip()
        except Exception:
            return None

        # Parse into dict
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
                    # Remove the prefix part in a case-insensitive way
                    # and store the remainder
                    value = ln[len(prefix):].strip(" :-")
                    out[key] = value
                    break

        return out

    def explain(self, label: str) -> Optional[str]:
        """
        High-level method used by the Streamlit app.

        Returns a short fused caption, e.g.:

        "Sust: Reduces labour but raises diesel use | Econ: Big CAPEX, long-run cost savings | Haz: PTO entanglement risk"
        """
        if not self.enabled or self.client is None:
            return None

        multi = self._call_openai(label)
        if not multi:
            return None

        # Compose a compact overlay string
        # Keep it short – we truncate in app anyway.
        pieces = []

        sust = multi.get("SUSTAINABILITY") or ""
        if sust:
            pieces.append(f"Sust: {sust}")

        econ = multi.get("ECONOMETRICS") or ""
        if econ:
            pieces.append(f"Econ: {econ}")

        hazard = multi.get("HAZARD") or ""
        if hazard:
            pieces.append(f"Haz: {hazard}")

        # You can comment these in/out depending on how dense you want the overlay
        supply = multi.get("SUPPLY_CHAIN") or ""
        if supply:
            pieces.append(f"SC: {supply}")

        lpis = multi.get("LPIS_GEO") or ""
        if lpis:
            pieces.append(f"LPIS: {lpis}")

        # Join and let the video processor cut to length
        fused = " | ".join(pieces)
        return fused if fused else None
