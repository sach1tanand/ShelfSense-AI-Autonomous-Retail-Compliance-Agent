"""Small OpenAI-compatible chat client for answering questions about shelf audits."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class ChatProviderError(RuntimeError):
    """The configured chat provider could not return a usable answer."""


def load_local_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from the project .env without overriding process env."""
    env_path = path or Path(__file__).resolve().parents[1] / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[7:].strip()
        name, separator, value = entry.partition("=")
        if not separator or not name.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)


load_local_env()


def chat_configured() -> bool:
    return bool(os.getenv("SHELFSENSE_CHAT_API_KEY", "").strip())


def audit_context(report_id: str | None, report: dict[str, Any] | None) -> dict[str, Any]:
    """Select only the audit facts needed for a chat response; never include image paths."""
    if not report:
        return {"report_id": None, "available": False}

    score = report.get("compliance_score") or {}
    comparison = report.get("planogram_comparison") or {}
    summary = comparison.get("summary") or {}
    return {
        "available": True,
        "compliance_score": {
            key: score.get(key) for key in (
                "score", "status", "coverage", "reason", "breakdown", "weights", "formula",
                "compliant_checks", "violation_checks", "unresolved_checks", "applicable_check_count",
            ) if key in score
        },
        "compliance_summary": {
            key: summary.get(key) for key in (
                "matches", "missing_products", "wrong_or_misplaced", "voids",
                "unexpected_products", "needs_review", "price_mismatches", "price_unreadable",
            ) if key in summary
        },
        "slots": [
            {
                key: slot.get(key) for key in (
                    "shelf", "row", "position", "slot", "sku", "expected_sku", "product",
                    "expected", "observed", "status", "reason_code", "detail",
                ) if key in slot
            }
            for slot in (report.get("compliance") or [])[:100]
        ],
        "findings": [
            {
                key: finding.get(key) for key in (
                    "shelf", "row", "slot", "expected_sku", "expected_product", "detected_sku",
                    "detected_product", "status", "suggested_status", "confirmed", "reason",
                    "price_status", "expected_price", "observed_price", "facing_count",
                ) if key in finding
            }
            for finding in (comparison.get("findings") or [])[:100]
        ],
        "price_checks": [
            {
                key: result.get(key) for key in (
                    "sku", "product", "expected_price", "observed_price", "status", "detail",
                ) if key in result
            }
            for result in (report.get("price_results") or [])[:100]
        ],
    }


def _history_messages(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    messages = []
    for item in history[-8:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            messages.append({"role": item["role"], "content": content.strip()[:1500]})
    return messages


def generate_reply(message: str, context: dict[str, Any], history: Any = None) -> str:
    api_key = os.getenv("SHELFSENSE_CHAT_API_KEY", "").strip()
    if not api_key:
        raise ChatProviderError("AI chat is not configured. Set SHELFSENSE_CHAT_API_KEY on the server and restart ShelfSense AI.")

    endpoint = os.getenv("SHELFSENSE_CHAT_API_URL", "https://api.openai.com/v1/chat/completions").strip()
    if urlparse(endpoint).scheme not in {"http", "https"} or not urlparse(endpoint).netloc:
        raise ChatProviderError("Chat provider URL must be an HTTP or HTTPS URL.")
    is_gemini = "generativelanguage.googleapis.com" in (urlparse(endpoint).hostname or "").lower()
    model = os.getenv(
        "SHELFSENSE_CHAT_MODEL", "gemini-3.5-flash-lite" if is_gemini else "gpt-4o-mini"
    ).strip()
    if is_gemini:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
            raise ChatProviderError("Gemini model name contains unsupported characters.")
        endpoint, replaced = re.subn(
            r"(/models/)[^/:]+(:generateContent)$", rf"\g<1>{model}\g<2>", endpoint, count=1
        )
        if not replaced:
            raise ChatProviderError("Gemini URL must end with /models/<model>:generateContent.")
    system_prompt = (
        "You are ShelfSense AI's assistant. Explain the application and answer questions about the supplied audit. "
        "Use plain language and concise, actionable guidance. Treat the audit JSON as the only source of audit facts. "
        "Never invent products, prices, scores, detections, or confirmed findings. Clearly label unconfirmed and "
        "Needs Review results. If the requested fact is absent, say the report does not contain it. Do not claim to "
        "have viewed an image. Explain that compliance score reflects the recorded checks and unresolved evidence. "
        "Format answers for readability using short headings, bullet lists, and bold labels where useful. For audit "
        "summaries, organize available facts under Overall status, Confirmed findings, Needs Review, and Recommended "
        "next steps; omit sections with no relevant data. Do not show raw Markdown syntax or repeat unsupported data. "
        "Known application workflow: upload a shelf photo, match detected products against the catalogue, compare "
        "them with a planogram, optionally verify shelf-label prices with OCR, review uncertain results, and download "
        "or revisit saved reports. Product identity and trusted shelf placement are separate checks."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "Current audit data (authoritative JSON): " + json.dumps(context, ensure_ascii=False)},
        *_history_messages(history),
        {"role": "user", "content": message.strip()[:1500]},
    ]
    if is_gemini:
        system_text = "\n\n".join(item["content"] for item in messages if item["role"] == "system")
        contents = [
            {
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": item["content"]}],
            }
            for item in messages if item["role"] != "system"
        ]
        request_body = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2},
        }
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    else:
        request_body = {"model": model, "messages": messages, "temperature": 0.2}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        provider_message = ""
        try:
            error_payload = json.loads(error.read().decode("utf-8"))
            provider_message = str((error_payload.get("error") or {}).get("message") or "").strip()
        except (AttributeError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
            pass
        if provider_message:
            provider_message = provider_message.replace(api_key, "[redacted]")
            provider_message = re.sub(r"(?i)(key\s*[=:]\s*)\S+", r"\1[redacted]", provider_message)
            provider_message = provider_message[:300]
            provider_message = f" Provider detail: {provider_message}"
        if error.code in {401, 403}:
            raise ChatProviderError(
                f"Chat provider rejected the API key (HTTP {error.code}). Check that the key is valid for "
                "the configured provider and has permission to use the selected model." + provider_message
            ) from error
        raise ChatProviderError(f"Chat provider returned HTTP {error.code}.{provider_message}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise ChatProviderError("Could not reach the chat provider before the request timed out or failed.") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ChatProviderError("Chat provider returned an invalid response.") from error

    try:
        if is_gemini:
            parts = result["candidates"][0]["content"]["parts"]
            answer = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        else:
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
            answer = str(content).strip()
    except (KeyError, IndexError, TypeError) as error:
        raise ChatProviderError("Chat provider response did not include an answer.") from error
    if not answer:
        raise ChatProviderError("Chat provider returned an empty answer.")
    return answer[:6000]
