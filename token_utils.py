"""Utility helpers for token counting and truncation."""
from __future__ import annotations

from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency at runtime
    import tiktoken
except ImportError:  # pragma: no cover - provide graceful fallback
    tiktoken = None  # type: ignore[assignment]


def _get_encoding(model: str):
    """Return the best-effort encoding for a given model."""

    if tiktoken is None:  # pragma: no cover - fallback path
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens_text(text: str, model: str) -> int:
    """Count tokens for raw text according to the model's encoding."""

    if not text:
        return 0
    encoding = _get_encoding(model)
    if encoding is None:  # pragma: no cover - fallback heuristic
        return max(1, len(text) // 4)
    return len(encoding.encode(text))


def truncate_context_text(text: str, model: str, max_tokens: int) -> str:
    """Truncate text so it fits within ``max_tokens`` for the given model."""

    if max_tokens <= 0 or not text:
        return ""
    encoding = _get_encoding(model)
    if encoding is None:  # pragma: no cover - fallback heuristic
        if len(text) <= max_tokens:
            return text
        return text[:max_tokens]
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def count_tokens_chat(messages: Sequence[Dict[str, str]], model: str) -> int:
    """Approximate the number of tokens consumed by chat messages."""

    if not messages:
        return 0

    encoding = _get_encoding(model)
    if encoding is None:  # pragma: no cover - fallback heuristic
        return sum(max(1, len(msg.get("content", "")) // 4 + 3) for msg in messages) + 3

    tokens_per_message = 3
    tokens_per_name = 1
    total = 0
    for message in messages:
        total += tokens_per_message
        total += len(encoding.encode(message.get("content", "") or ""))
        if message.get("name"):
            total += tokens_per_name
    total += 3
    return total


def truncate_messages_to_budget(
    messages: Sequence[Dict[str, str]],
    model: str,
    max_input_tokens: int,
    reserve_output_tokens: int = 1024,
) -> List[Dict[str, str]]:
    """Trim the conversation so it fits under the token budget."""

    if not messages or max_input_tokens <= 0:
        return []

    budget = max(max_input_tokens - max(reserve_output_tokens, 0), 0)
    if budget == 0:
        return []

    system_messages: List[Dict[str, str]] = [dict(msg) for msg in messages if msg.get("role") == "system"]
    other_messages: List[Dict[str, str]] = [dict(msg) for msg in messages if msg.get("role") != "system"]

    kept_reversed: List[Dict[str, str]] = []
    for message in reversed(other_messages):
        candidate = list(reversed(kept_reversed + [message]))
        candidate_messages = system_messages + candidate
        if count_tokens_chat(candidate_messages, model) <= budget or not kept_reversed:
            kept_reversed.append(message)
        else:
            break

    trimmed = system_messages + list(reversed(kept_reversed))
    total_tokens = count_tokens_chat(trimmed, model)

    # Remove oldest non-system messages until we meet the budget or only keep the latest.
    while total_tokens > budget and len(trimmed) > 1:
        first_non_system = next(
            (idx for idx, msg in enumerate(trimmed) if msg.get("role") != "system"),
            None,
        )
        if first_non_system is None or first_non_system == len(trimmed) - 1:
            break
        trimmed.pop(first_non_system)
        total_tokens = count_tokens_chat(trimmed, model)

    if trimmed and total_tokens > budget:
        # Truncate the most recent non-system message to squeeze within the budget.
        target_index = next(
            (idx for idx in range(len(trimmed) - 1, -1, -1) if trimmed[idx].get("role") != "system"),
            len(trimmed) - 1,
        )
        target = dict(trimmed[target_index])
        prefix = trimmed[:target_index]
        suffix = trimmed[target_index + 1 :]

        encoding = _get_encoding(model)
        if encoding is None:  # pragma: no cover - fallback heuristic
            content = target.get("content", "")
            allowed = max(budget - count_tokens_chat(prefix + suffix, model), 0)
            target["content"] = content[:allowed] if allowed < len(content) else content
        else:
            content_tokens = encoding.encode(target.get("content", "") or "")
            low, high = 0, len(content_tokens)
            best_tokens = []
            while low <= high:
                mid = (low + high) // 2
                candidate_tokens = content_tokens[:mid]
                candidate_content = encoding.decode(candidate_tokens) if candidate_tokens else ""
                candidate_messages = prefix + [dict(target, content=candidate_content)] + suffix
                if count_tokens_chat(candidate_messages, model) <= budget:
                    best_tokens = candidate_tokens
                    low = mid + 1
                else:
                    high = mid - 1
            target["content"] = (
                encoding.decode(best_tokens) if best_tokens else ""
            )
        trimmed = prefix + [target] + suffix

    return trimmed
