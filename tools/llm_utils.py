import json
from langchain_core.messages import BaseMessage


def extract_text(response: BaseMessage) -> str:
    """Extract plain text from an LLM response regardless of provider content format."""
    content = response.content
    if isinstance(content, str):
        return content
    # Gemini (and some other providers) return a list of content parts
    parts = []
    for part in content:
        if isinstance(part, dict):
            parts.append(part.get("text", ""))
        else:
            parts.append(str(part))
    return "".join(parts)


def parse_json_response(response: BaseMessage, agent_name: str) -> dict | list:
    """Extract text from a response and parse it as JSON, stripping code fences."""
    raw = extract_text(response).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[{agent_name}] Failed to parse LLM response as JSON: {e}\nRaw response: {raw}"
        ) from e
