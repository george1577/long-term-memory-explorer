"""Getting JSON out of model replies, which small models don't always format cleanly."""

from __future__ import annotations

import json
import re


def parse_json_object(text: str) -> dict | None:
    """Parse a JSON object from a reply. Handles ```json fences and chatter around the object.

    Returns None if nothing parseable is found (the pipeline then treats it as "no facts" or
    "no decisions", the same way mem0 does).
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
